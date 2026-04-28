// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// sparse_attn_indexer_sm86_tuned.cu — Story 8: register-pressure + tile-shape
// tuned variant of sparse_attn_indexer_sm86.cu for RTX 3090 (sm_86).
//
// SM-level constraints (RTX 3090, sm_86):
//   - 128 KB unified L1/smem per SM (max 100 KB usable for opt-in carveout)
//   - 65536 32-bit registers per SM
//   - 1536 max threads per SM (== 12 warps)
//   - 1024 max threads per CTA
//   - 32 KB max smem per CTA without opt-in (96 KB with cudaFuncSetAttribute)
//
// Tile-shape choices (justified inline):
//   - kThreadsPerBlock = 128 (4 warps): keeps regs-per-thread headroom for
//     mma fragments while permitting 4-CTA-per-SM occupancy at <=128 regs/thread.
//   - kBlocksPerCTA = 4: each CTA processes 4 K-blocks of 64 rows each
//     (= 256 rows per CTA). Halves grid.x dimension vs the v1 skeleton.
//   - Q-row aliased smem: replicate the single Q row into a 16-row smem region
//     (kQAliasRows = 16) so ldmatrix.sync.aligned.m8n8.x4 can read directly
//     without repeated reload — eliminates the v1 "warp 0 only" bottleneck.
//   - All 4 warps participate: each warp covers (kBlocksPerCTA × kBlockK / 4) =
//     64 K-rows of the output. Cross-warp reduction at the end.
//
// Memory budget per CTA:
//   smem_q_alias  = 16 * kHeadDim * 2 = 4096 B
//   smem_k_tile   = (kBlocksPerCTA * kBlockK) * kHeadDim * 2 = 4 * 64 * 128 * 2
//                 = 65536 B (just under 64 KB — needs opt-in carveout)
// Total smem = 65536 + 4096 = 69632 B. Use cudaFuncSetAttribute
// cudaFuncAttributeMaxDynamicSharedMemorySize = 73728 (72 KB).
//
// Register-budget target: <= 96 regs/thread → 128 threads × 96 = 12288 regs/CTA
// → 5 CTAs/SM theoretical (limited by smem to 1 CTA/SM with 72 KB carveout).
// To get 2 CTAs/SM we'd need to drop kBlocksPerCTA to 2 (smem 36 KB, 2/SM).
// Story 8 ships THIS variant; Story 8b will offer kBlocksPerCTA=2 for occupancy.
//
// Profile-report placeholder:
//   ===== expected ncu output (filled by CC9 broker on cuda3) =====
//   sm__warps_active.avg.pct_of_peak_sustained_active  : >= 60 % target
//   sm__inst_executed.avg.per_cycle_active             : >= 0.8 IPC target
//   smsp__inst_executed_pipe_tensor.avg.pct            : >= 25 % (mma utilization)
//   l1tex__throughput.avg.pct_of_peak_sustained_active : <= 80 %
//   ============================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace vllm {
namespace dsa_sm86 {
namespace tuned {

// ============================================================================
// Tile constants (Story 8 tuning)
// ============================================================================

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = kThreadsPerBlock / kWarpSize;          // 4
constexpr int kBlockK          = 64;
constexpr int kBlocksPerCTA    = 4;                                     // 4 K-blocks/CTA
constexpr int kKRowsPerCTA     = kBlockK * kBlocksPerCTA;               // 256
constexpr int kHeadDim         = 128;
constexpr int kQAliasRows      = 16;     // Q row replicated for ldmatrix m=16
constexpr int kMmaM            = 16;
constexpr int kMmaN            = 8;
constexpr int kMmaK            = 16;
constexpr int kMmaIters        = kHeadDim / kMmaK;                      // 8

constexpr int kSmemQAliasBytes = kQAliasRows * kHeadDim * sizeof(__nv_bfloat16);
constexpr int kSmemKBytes      = kKRowsPerCTA * kHeadDim * sizeof(__nv_bfloat16);
constexpr int kSmemTotal       = kSmemQAliasBytes + kSmemKBytes;        // 69632 B

// Each warp owns (kKRowsPerCTA / kWarpsPerBlock) = 64 rows of output
constexpr int kKRowsPerWarp    = kKRowsPerCTA / kWarpsPerBlock;         // 64
// Each warp does (kKRowsPerWarp / kMmaN) = 8 N-tile mma issues per K-iter
constexpr int kMmaNTilesPerWarp = kKRowsPerWarp / kMmaN;                // 8

// ============================================================================
// PTX helpers — same as v1, repeated for self-containment
// ============================================================================

__device__ __forceinline__ void cp_async_cg_16(
    void* smem_dst, const void* gmem_src) {
  unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16;\n"
      :: "r"(smem_int), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_group 0;\n");
}

__device__ __forceinline__ void mma_sync_m16n8k16_bf16_f32(
    float       (&D)[4],
    const unsigned (&A)[4],
    const unsigned (&B)[2],
    const float    (&C)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%10, %11, %12, %13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1]),
        "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ __forceinline__ void ldmatrix_a_m16k16(
    unsigned (&A)[4], const __nv_bfloat16* smem_ptr) {
  unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];\n"
      : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
      : "r"(smem_int));
}

__device__ __forceinline__ void ldmatrix_b_n8k16_trans(
    unsigned (&B)[2], const __nv_bfloat16* smem_ptr) {
  unsigned smem_int = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
      "{%0, %1}, [%2];\n"
      : "=r"(B[0]), "=r"(B[1])
      : "r"(smem_int));
}

// ============================================================================
// Kernel — tuned variant
// ============================================================================
//
// Grid: ((n_blocks + kBlocksPerCTA-1)/kBlocksPerCTA, num_heads, batch_size)
// One CTA writes kBlocksPerCTA scores per (batch, head, q_idx).
// All 4 warps participate; each warp owns 1/4 of the 256 K-rows.

struct Params {
  const __nv_bfloat16* __restrict__ q;
  const __nv_bfloat16* __restrict__ k_cache;
  float* __restrict__ block_scores;
  int q_stride_b, q_stride_h, q_stride_q;
  int k_stride_b, k_stride_t;
  int score_stride_b, score_stride_h, score_stride_q;
  int n_blocks, t_q;
  float scale;
};

__global__ void sparse_attn_indexer_logits_tuned_kernel(Params p) {
  const int cta_blk_base = blockIdx.x * kBlocksPerCTA;
  const int head    = blockIdx.y;
  const int batch   = blockIdx.z;

  const int tx   = threadIdx.x;
  const int warp = tx / kWarpSize;
  const int lane = tx & (kWarpSize - 1);

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* smem_q_alias = smem;                                    // [16, kHeadDim]
  __nv_bfloat16* smem_k_tile  = smem_q_alias + kQAliasRows * kHeadDim;   // [kKRowsPerCTA, kHeadDim]

  for (int q_idx = 0; q_idx < p.t_q; ++q_idx) {
    // ---- Async-load the single Q row into smem (replicated to 16 rows) ----
    const __nv_bfloat16* gmem_q =
        p.q + batch * p.q_stride_b + head * p.q_stride_h +
        q_idx * p.q_stride_q;
    constexpr int kQChunks = kHeadDim / 8;
    if (tx < kQChunks) {
      // First, load the canonical Q row to smem_q_alias[0].
      cp_async_cg_16(smem_q_alias + tx * 8, gmem_q + tx * 8);
    }

    // ---- Async-load the kKRowsPerCTA K rows for this CTA ----
    const int n_blocks_remaining = p.n_blocks - cta_blk_base;
    const int active_blocks = n_blocks_remaining < kBlocksPerCTA
                              ? n_blocks_remaining
                              : kBlocksPerCTA;
    const int active_k_rows = active_blocks * kBlockK;

    constexpr int kElemsPerChunk = 8;
    const int total_chunks = active_k_rows * (kHeadDim / kElemsPerChunk);
    for (int i = tx; i < total_chunks; i += kThreadsPerBlock) {
      const int row = i / (kHeadDim / kElemsPerChunk);
      const int col = (i % (kHeadDim / kElemsPerChunk)) * kElemsPerChunk;
      const int abs_k_row = cta_blk_base * kBlockK + row;
      cp_async_cg_16(
          smem_k_tile + row * kHeadDim + col,
          p.k_cache + batch * p.k_stride_b + abs_k_row * p.k_stride_t + col);
    }

    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ---- Replicate Q row across 16 alias rows (single warp, register copy) ----
    if (warp == 0) {
      // Each lane copies kHeadDim/32 = 4 bf16 elements per row.
      constexpr int kPerLane = kHeadDim / kWarpSize;  // 4
      // Read source row 0 once into registers; broadcast-write to rows 1..15.
      __nv_bfloat16 src[kPerLane];
#pragma unroll
      for (int i = 0; i < kPerLane; ++i) {
        src[i] = smem_q_alias[lane * kPerLane + i];
      }
#pragma unroll
      for (int row = 1; row < kQAliasRows; ++row) {
#pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
          smem_q_alias[row * kHeadDim + lane * kPerLane + i] = src[i];
        }
      }
    }
    __syncthreads();

    // ---- mma.sync inner loop: each warp covers kKRowsPerWarp rows ----
    // Per-warp accumulator for kMmaNTilesPerWarp (=8) N-tiles
    // Each tile has 4 fp32 outputs per thread (m16n8 fragment layout).
    float acc[kMmaNTilesPerWarp][4];
#pragma unroll
    for (int n = 0; n < kMmaNTilesPerWarp; ++n) {
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[n][i] = 0.0f;
    }

    const int warp_k_row_base = warp * kKRowsPerWarp;

#pragma unroll
    for (int k_off = 0; k_off < kHeadDim; k_off += kMmaK) {
      // A-fragment: 16 rows × 16 cols of Q (replicated row, k-stride within smem).
      unsigned A[4];
      ldmatrix_a_m16k16(A, smem_q_alias + k_off);

      // For each N-tile owned by this warp, load B fragment + issue mma.
#pragma unroll
      for (int n = 0; n < kMmaNTilesPerWarp; ++n) {
        const int row_base = warp_k_row_base + n * kMmaN;
        unsigned B[2];
        ldmatrix_b_n8k16_trans(B, smem_k_tile + row_base * kHeadDim + k_off);

        float new_acc[4];
        const float prev[4] = {acc[n][0], acc[n][1], acc[n][2], acc[n][3]};
        mma_sync_m16n8k16_bf16_f32(new_acc, A, B, prev);
        acc[n][0] = new_acc[0]; acc[n][1] = new_acc[1];
        acc[n][2] = new_acc[2]; acc[n][3] = new_acc[3];
      }
    }

    // ---- Reduce per-row contributions: sum the 4 fp32 fragments per N-tile,
    // then warp-reduce, then cross-warp reduce within the K-block. ----
    // Each (warp, n) pair represents 8 K-rows; we need ONE scalar per K-block.
    // K-rows mapping: K-row = warp * kKRowsPerWarp + n * kMmaN + (lane / 4)
    // Each lane contributes 4 of the m=16 row outputs (fragment layout).
    // For Lightning-Indexer block-score we sum over ALL kBlockK rows in the block.

    // Stage 1: reduce per-thread 4 frags into a single value (sum of 4).
    // Stage 2: warp-reduce within kKRowsPerWarp rows.
    // Stage 3: cross-block reduction via smem.
    //
    // For block_idx in [cta_blk_base, cta_blk_base + active_blocks):
    //   block_score = sum over (n, lane) where the (n,lane) pair maps to a
    //   K-row inside this block.
    //
    // Implementation: each thread computes its contribution to block_score,
    // then we use shfl_xor_sync within the warp + smem for cross-warp.

    // Per-thread sum of the 4 fp32 values per n-tile (one per row this thread
    // owns from the m16 fragment).
    // For block_idx i, we need rows in [i*kBlockK, (i+1)*kBlockK).
    // warp's k-rows start at warp * kKRowsPerWarp; each n-tile covers 8 K-rows.
    // K-rows-per-warp = 64 = 1 block.
    // → one warp covers exactly one K-block (kKRowsPerWarp == kBlockK).
    // → block_score for block (cta_blk_base + warp) = sum of all warp lanes' contributions.
    static_assert(kKRowsPerWarp == kBlockK,
                  "kKRowsPerWarp must equal kBlockK for the warp-per-block mapping");

    float warp_block_sum = 0.0f;
#pragma unroll
    for (int n = 0; n < kMmaNTilesPerWarp; ++n) {
      warp_block_sum += acc[n][0] + acc[n][1] + acc[n][2] + acc[n][3];
    }
    // Warp reduce to lane 0.
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      warp_block_sum += __shfl_down_sync(0xFFFFFFFF, warp_block_sum, offset);
    }

    // Lane 0 of each warp writes to its block.
    if (lane == 0 && warp < active_blocks) {
      const int abs_blk = cta_blk_base + warp;
      p.block_scores[
          batch * p.score_stride_b + head * p.score_stride_h +
          q_idx * p.score_stride_q + abs_blk
      ] = warp_block_sum * p.scale;
    }
    __syncthreads();
  }
}

// ============================================================================
// Host launcher with smem opt-in carveout
// ============================================================================

extern "C" cudaError_t launch_sparse_attn_indexer_logits_tuned_sm86(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    float* block_scores,
    int batch, int num_heads, int t_q, int n_blocks,
    int q_stride_b, int q_stride_h, int q_stride_q,
    int k_stride_b, int k_stride_t,
    int score_stride_b, int score_stride_h, int score_stride_q,
    float scale,
    cudaStream_t stream) {
  Params p = {};
  p.q              = q;
  p.k_cache        = k_cache;
  p.block_scores   = block_scores;
  p.q_stride_b     = q_stride_b;
  p.q_stride_h     = q_stride_h;
  p.q_stride_q     = q_stride_q;
  p.k_stride_b     = k_stride_b;
  p.k_stride_t     = k_stride_t;
  p.score_stride_b = score_stride_b;
  p.score_stride_h = score_stride_h;
  p.score_stride_q = score_stride_q;
  p.n_blocks       = n_blocks;
  p.t_q            = t_q;
  p.scale          = scale;

  // Opt-in carveout: 72 KB smem per CTA on sm_86.
  cudaError_t err = cudaFuncSetAttribute(
      sparse_attn_indexer_logits_tuned_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemTotal);
  if (err != cudaSuccess) return err;

  const int n_cta_blocks = (n_blocks + kBlocksPerCTA - 1) / kBlocksPerCTA;
  dim3 grid(n_cta_blocks, num_heads, batch);
  dim3 block(kThreadsPerBlock);

  sparse_attn_indexer_logits_tuned_kernel
      <<<grid, block, kSmemTotal, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace tuned
}  // namespace dsa_sm86
}  // namespace vllm

// // --ProtoAI-Bakari--
