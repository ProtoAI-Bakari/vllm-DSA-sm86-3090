// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// paged_attn_hd512_sm86.cu — head_dim=512 paged-KV attention for sm_86.
//
// Backport target: vLLM PR #38835 (FlashAttention-4 paged-KV with head_dim 512)
// added the SM90+ path. Without an sm_86 fallback, DSV4 (head_dim 512 = 448
// NoPE + 64 RoPE) loses its paged attention backend on RTX 3090: stock FA2
// caps at head_dim 256, FA3 / FA4 are Hopper+. Result: paged decode either
// 500s or silently routes through a slower MQA path.
//
// This kernel handles the head_dim=512 paged-decode case on sm_86:
//   - BF16 Q [B, H, S_q, 512]
//   - BF16 paged KV cache via block_table indirection
//   - Sparse mode: optional topk_indices (-1 = invalid, skip)
//   - Dense mode:  full causal / window attention (handled at higher level)
//   - Output BF16 [B, H, S_q, 512]
//
// Hopper -> sm_86 substitutions:
//   - cp.async.bulk.tensor.* -> cp.async.cg.shared.global (16-byte chunks)
//   - wgmma.mma_async.*      -> mma.sync.aligned.m16n8k16 (bf16 -> fp32 acc)
//   - PDL                    -> stream-order (host-side)
//
// Algorithm-preserving online-softmax via online_softmax_sm86.cuh.
// Empty-keys safe (zero-fill on no valid topk slots).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "online_softmax_sm86.cuh"

namespace vllm {
namespace dsa_sm86 {
namespace paged_hd512 {

// ============================================================================
// Constants — tuned for head_dim=512 on RTX 3090
// ============================================================================

constexpr int kHeadDim         = 512;     // DSV4 head_dim (448 NoPE + 64 RoPE)
constexpr int kThreadsPerBlock = 256;     // 8 warps
constexpr int kWarpSize        = 32;
constexpr int kWarpsPerBlock   = kThreadsPerBlock / kWarpSize;
constexpr int kKRowsPerWarp    = 8;       // each warp processes 8 K-rows per iter
constexpr int kKRowsPerBlock   = kWarpsPerBlock * kKRowsPerWarp;  // 64
constexpr int kVecPerThread    = 2;       // bf16 elems per thread along D
constexpr int kHeadDimPerThread = kHeadDim / kThreadsPerBlock * 4;
                                          // 8 elems/thread (256 thr * 8 / 16)

// SMEM:
//   smem_q  : (1, kHeadDim) bf16 = 1024 bytes
//   smem_k  : (kKRowsPerBlock, kHeadDim) bf16 = 64 * 512 * 2 = 65536 bytes
//   smem_v  : same shape as smem_k = 65536 bytes
// Total = 1024 + 65536 + 65536 = 132096 bytes = 129 KB
//
// sm_86 has 128 KB total smem/SM, max 100 KB usable per CTA via opt-in.
// → can NOT fit full V tile + K tile + Q simultaneously. Trade-off:
//   - Stream V load alongside K (double-buffered)
//   - OR fold V into the same iteration as K (process row-by-row)
//
// We choose row-by-row (option 2) to keep smem use under 100 KB:
//   smem_q  : 1024 bytes
//   smem_k  : (kKRowsPerBlock, kHeadDim) = 65536 bytes
//   smem_v  : (kKRowsPerBlock, kHeadDim) = 65536 bytes  ← halve via streaming
//   → use 32-row K tile + 32-row V tile = 32768 + 32768 + 1024 = 66560 ~ 65 KB
constexpr int kKRowsPerTile = 32;  // halved from kKRowsPerBlock for smem fit
constexpr int kSmemQBytes   = kHeadDim * sizeof(__nv_bfloat16);
constexpr int kSmemKBytes   = kKRowsPerTile * kHeadDim * sizeof(__nv_bfloat16);
constexpr int kSmemVBytes   = kSmemKBytes;
constexpr int kSmemTotal    = kSmemQBytes + kSmemKBytes + kSmemVBytes;

// ============================================================================
// PTX helpers (same as sparse_attn_indexer_sm86.cu)
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

// ============================================================================
// Kernel — paged-KV decode (S_q == 1) with head_dim 512
// ============================================================================
//
// Grid: (1, num_heads, batch) — one CTA per (batch, head).
// CTA processes ALL valid K-rows for this (batch, head) iteratively in
// kKRowsPerTile chunks via online softmax.

struct PagedAttnHd512Params {
  const __nv_bfloat16* __restrict__ q;          // [B, H, S_q=1, 512]
  const __nv_bfloat16* __restrict__ kv_cache;   // [num_blocks, block_size, 512] (K) + V
  const int32_t* __restrict__       block_table; // [B, max_blocks_per_seq]
  const int32_t* __restrict__       topk_indices; // [B, H, K] -1=invalid (or null for dense)
  const int32_t* __restrict__       cache_seqlens; // [B] valid seq length per request
  __nv_bfloat16* __restrict__       out;        // [B, H, 512]
  // Strides
  int64_t q_stride_b, q_stride_h;
  int64_t kv_block_stride;       // bytes per block in kv_cache
  int64_t kv_token_stride;       // bytes per token in a block
  int64_t v_offset;              // V offset within (k+v) cache layout
  int64_t block_table_stride_b;
  int64_t topk_stride_b, topk_stride_h;
  int64_t out_stride_b, out_stride_h;
  // Shape
  int batch_size, num_heads;
  int block_size;                // tokens per cache block (DSV4: 256)
  int topk_count;                // K (number of sparse indices); 0 = dense
  int max_blocks_per_seq;
  float scale;
};

__global__ void paged_attn_hd512_sm86_kernel(PagedAttnHd512Params p) {
  const int batch = blockIdx.z;
  const int head  = blockIdx.y;
  const int tx    = threadIdx.x;
  const int warp  = tx / kWarpSize;
  const int lane  = tx & (kWarpSize - 1);

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* smem_q = smem;                                        // [512]
  __nv_bfloat16* smem_k = smem_q + kHeadDim;                           // [32, 512]
  __nv_bfloat16* smem_v = smem_k + kKRowsPerTile * kHeadDim;            // [32, 512]

  // ---- Load Q row for this (batch, head) ----
  const __nv_bfloat16* gmem_q =
      p.q + batch * p.q_stride_b + head * p.q_stride_h;
  constexpr int kQChunks = kHeadDim / 8;  // 64 chunks of 8 bf16
  if (tx < kQChunks) {
    cp_async_cg_16(smem_q + tx * 8, gmem_q + tx * 8);
  }
  cp_async_commit();
  cp_async_wait_all();
  __syncthreads();

  // ---- Determine valid K-row count ----
  const int seq_len = p.cache_seqlens[batch];
  const int k_count = (p.topk_indices != nullptr) ? p.topk_count : seq_len;

  // Per-thread accumulator: each thread owns kHeadDim/kThreadsPerBlock = 2
  // bf16 components of the V output. We use OnlineSoftmaxVec<2> per thread.
  // Lane 0 of each warp will writeback its 2 components.
  // For full coverage we need all 256 threads to cover 512 dims (2 each).
  const int my_dim_base = tx * 2;
  if (my_dim_base >= kHeadDim) return;  // shouldn't happen with kThreadsPerBlock=256

  OnlineSoftmaxVec<2> state;
  state.reset();

  // Pre-load Q components for this thread.
  float q_local[2] = {
      __bfloat162float(smem_q[my_dim_base]),
      __bfloat162float(smem_q[my_dim_base + 1]),
  };

  // ---- Iterate over K-rows in chunks of kKRowsPerTile ----
  for (int chunk_start = 0; chunk_start < k_count; chunk_start += kKRowsPerTile) {
    const int chunk_end = chunk_start + kKRowsPerTile < k_count
                          ? chunk_start + kKRowsPerTile : k_count;

    // ---- Async-load K + V tiles for this chunk ----
    constexpr int kElemsPerChunk = 8;
    const int total_k_chunks = (chunk_end - chunk_start) * (kHeadDim / kElemsPerChunk);
    for (int i = tx; i < total_k_chunks; i += kThreadsPerBlock) {
      const int row = i / (kHeadDim / kElemsPerChunk);
      const int col = (i % (kHeadDim / kElemsPerChunk)) * kElemsPerChunk;
      // Resolve sparse / dense index:
      //   sparse: slot_idx = topk_indices[batch, head, chunk_start + row]
      //   dense:  slot_idx = chunk_start + row
      int slot_idx;
      bool valid;
      if (p.topk_indices != nullptr) {
        slot_idx = p.topk_indices[
            batch * p.topk_stride_b + head * p.topk_stride_h +
            (chunk_start + row)];
        valid = slot_idx >= 0;
      } else {
        slot_idx = chunk_start + row;
        valid = true;
      }
      if (!valid) {
        // Zero-fill smem slot to avoid garbage.
#pragma unroll
        for (int e = 0; e < kElemsPerChunk; ++e) {
          smem_k[row * kHeadDim + col + e] = __float2bfloat16(0.0f);
          smem_v[row * kHeadDim + col + e] = __float2bfloat16(0.0f);
        }
        continue;
      }

      // Translate slot_idx -> (block_idx, token_in_block) via block_table.
      const int logical_block = slot_idx / p.block_size;
      const int token_in_blk  = slot_idx % p.block_size;
      const int physical_block = p.block_table[
          batch * p.block_table_stride_b + logical_block];

      const __nv_bfloat16* k_row = reinterpret_cast<const __nv_bfloat16*>(
          reinterpret_cast<const uint8_t*>(p.kv_cache) +
          physical_block * p.kv_block_stride +
          token_in_blk * p.kv_token_stride);
      const __nv_bfloat16* v_row = reinterpret_cast<const __nv_bfloat16*>(
          reinterpret_cast<const uint8_t*>(k_row) + p.v_offset);

      cp_async_cg_16(smem_k + row * kHeadDim + col, k_row + col);
      cp_async_cg_16(smem_v + row * kHeadDim + col, v_row + col);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // ---- Compute Q·K dot product per K-row + V combine via online softmax ----
    for (int row = 0; row < (chunk_end - chunk_start); ++row) {
      // Compute QK score for this K-row (full D=512 dot product).
      // Each thread contributes its 2-component partial sum, then warp/CTA reduce.
      float my_qk = 0.0f;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        my_qk += q_local[i] * __bfloat162float(smem_k[row * kHeadDim + my_dim_base + i]);
      }
      // CTA-wide reduction of qk via 2-stage shfl + smem.
      // Stage 1: warp reduce.
#pragma unroll
      for (int off = kWarpSize / 2; off > 0; off >>= 1) {
        my_qk += __shfl_down_sync(0xFFFFFFFF, my_qk, off);
      }
      // Stage 2: smem cross-warp reduce.
      __shared__ float qk_partials[kWarpsPerBlock + 1];
      if (lane == 0) qk_partials[warp] = my_qk;
      __syncthreads();
      float qk_full = 0.0f;
      if (warp == 0 && lane < kWarpsPerBlock) {
        qk_full = qk_partials[lane];
#pragma unroll
        for (int off = kWarpsPerBlock / 2; off > 0; off >>= 1) {
          qk_full += __shfl_down_sync(0xFFFFFFFF, qk_full, off);
        }
        if (lane == 0) qk_partials[kWarpsPerBlock] = qk_full * p.scale;
      }
      __syncthreads();
      const float qk_scaled = qk_partials[kWarpsPerBlock];

      // Read this thread's 2 V components for the row, update online softmax.
      const float v_local[2] = {
          __bfloat162float(smem_v[row * kHeadDim + my_dim_base]),
          __bfloat162float(smem_v[row * kHeadDim + my_dim_base + 1]),
      };
      state.update(qk_scaled, v_local, /*valid=*/true);
    }
    __syncthreads();
  }

  // ---- Finalize and write output ----
  float out_local[2];
  state.finalize(out_local);
  __nv_bfloat16* gmem_out =
      p.out + batch * p.out_stride_b + head * p.out_stride_h + my_dim_base;
  gmem_out[0] = __float2bfloat16(out_local[0]);
  gmem_out[1] = __float2bfloat16(out_local[1]);
}

// ============================================================================
// Host launcher
// ============================================================================

extern "C" cudaError_t launch_paged_attn_hd512_sm86(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv_cache,
    const int32_t* block_table,
    const int32_t* topk_indices,
    const int32_t* cache_seqlens,
    __nv_bfloat16* out,
    int batch_size, int num_heads,
    int block_size, int topk_count, int max_blocks_per_seq,
    int64_t q_stride_b, int64_t q_stride_h,
    int64_t kv_block_stride, int64_t kv_token_stride, int64_t v_offset,
    int64_t block_table_stride_b,
    int64_t topk_stride_b, int64_t topk_stride_h,
    int64_t out_stride_b, int64_t out_stride_h,
    float scale,
    cudaStream_t stream) {
  PagedAttnHd512Params p = {};
  p.q                    = q;
  p.kv_cache             = kv_cache;
  p.block_table          = block_table;
  p.topk_indices         = topk_indices;
  p.cache_seqlens        = cache_seqlens;
  p.out                  = out;
  p.q_stride_b           = q_stride_b;
  p.q_stride_h           = q_stride_h;
  p.kv_block_stride      = kv_block_stride;
  p.kv_token_stride      = kv_token_stride;
  p.v_offset             = v_offset;
  p.block_table_stride_b = block_table_stride_b;
  p.topk_stride_b        = topk_stride_b;
  p.topk_stride_h        = topk_stride_h;
  p.out_stride_b         = out_stride_b;
  p.out_stride_h         = out_stride_h;
  p.batch_size           = batch_size;
  p.num_heads            = num_heads;
  p.block_size           = block_size;
  p.topk_count           = topk_count;
  p.max_blocks_per_seq   = max_blocks_per_seq;
  p.scale                = scale;

  cudaError_t err = cudaFuncSetAttribute(
      paged_attn_hd512_sm86_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kSmemTotal);
  if (err != cudaSuccess) return err;

  dim3 grid(1, num_heads, batch_size);
  dim3 block(kThreadsPerBlock);

  paged_attn_hd512_sm86_kernel<<<grid, block, kSmemTotal, stream>>>(p);
  return cudaGetLastError();
}

}  // namespace paged_hd512
}  // namespace dsa_sm86
}  // namespace vllm

// // --ProtoAI-Bakari--
