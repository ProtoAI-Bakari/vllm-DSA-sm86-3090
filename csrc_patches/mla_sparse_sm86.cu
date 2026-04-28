// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to ProtoAI-Bakari/vllm-DSA-sm86-3090
//
// CC4 Story 6 — FlashMLA-Sparse decode kernel, sm_86 port.
//
// Replaces the Hopper-only `model1_persistent_h{64,128}.cu` from upstream
// FlashMLA. This is a NON-persistent, single-CTA-per-(batch,head) kernel
// that compiles + runs on sm_80 / sm_86 / sm_89. No TMA, no WGMMA, no PDL.
//
// Algorithm — sparse online-softmax MLA decode:
//   inputs:
//     Q          [B, H, D]  bf16  (D = 512 for DSV4 head_size, RoPE in last 64)
//     KV cache   paged 584B/token (per `flashmla_sparse.py:81-89`):
//                  [0..448)    : 448 fp8e4m3 NoPE bytes
//                  [448..576)  : 64 bf16 RoPE values (128 bytes)
//                  [576..584)  : 7 ue8m0 scale bytes + 1 pad
//     topk_idx   [B, H, K]  int32, valid index into the paged cache (-1 = invalid)
//     block_table[B, max_blocks] int32 — paged → physical block mapping
//   output:
//     out        [B, H, Dv] bf16   (Dv = 512 — same as D for MLA)
//
// Per (b, h), one CTA:
//   1. Load Q row (D=512 bf16) once into smem.
//   2. For each k in 0..K-1 (topk):
//        - load K_idx = topk_idx[b, h, k]
//        - if K_idx < 0: skip (early-exit when whole K row is invalid)
//        - dequant FP8 NoPE → bf16 (per-block-of-64 ue8m0 scale)
//        - copy bf16 RoPE 64 lanes
//        - QK^T via mma.sync.aligned.m16n8k16 → fp32 score
//        - online-softmax accumulator update
//        - V is the same K row in MLA (latent space); accumulate score * V
//   3. Normalize by softmax denominator, store out row in bf16.
//
// Performance target (Story 10 v2 will tune; this is correctness-first):
//   ~30-50% of Hopper FlashMLA throughput on RTX 3090 — bandwidth bound.
//
// External symbol from CC3 lane (lane-cc3-sparse-attn-cuda):
//   void v4_fp8_kv_to_bf16_sm86(const uint8_t* token584, __nv_bfloat16* k_bf16);
//   Dequants one 584B token to 512 bf16 (NoPE 448 → fp32 → bf16 + RoPE 64 bf16 passthrough).
//
// // --ProtoAI-Bakari--

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <torch/extension.h>

namespace cc4_mla_sparse_sm86 {

// =============================================================================
// External dequant from CC3 lane
// =============================================================================
extern __device__ void v4_fp8_kv_to_bf16_sm86(
    const uint8_t* __restrict__ token584,
    __nv_bfloat16* __restrict__ k_bf16);

// =============================================================================
// Constants — match upstream DSV4 layout
// =============================================================================
constexpr int kHeadDim   = 512;          // D (also Dv for MLA latent)
constexpr int kRopeDim   = 64;           // last 64 of D are RoPE-applied bf16
constexpr int kNopeDim   = kHeadDim - kRopeDim;  // 448 fp8 (DSV4)
constexpr int kTokenBytes = 584;         // per-token KV cache footprint

// MMA tile sizes for m16n8k16 (sm_80+ bf16 → fp32)
constexpr int kMmaM = 16;
constexpr int kMmaN = 8;
constexpr int kMmaK = 16;

// CTA layout: 4 warps × 32 threads. One CTA = one (batch, head) tile.
constexpr int kWarpsPerCTA  = 4;
constexpr int kThreadsPerCTA = kWarpsPerCTA * 32;

// Smem budget: Q row (1024B) + K-rolling-tile (4 K rows × 1024B = 4096B)
//            + softmax stats (8 × kMmaM lanes × fp32 = 512B) + scratch
constexpr int kKRollWindow = 4;
constexpr int kSmemBytes   = kHeadDim * sizeof(__nv_bfloat16)        // Q
                           + kKRollWindow * kHeadDim * sizeof(__nv_bfloat16) // K
                           + kMmaM * sizeof(float) * 4               // stats
                           + 256;                                    // pad

// =============================================================================
// MMA wrapper — m16n8k16 bf16×bf16=fp32, sm_80 PTX
// =============================================================================
__device__ __forceinline__ void mma_m16n8k16_bf16_fp32(
    float (&D)[4],
    const uint32_t (&A)[4],   // 4 × 32-bit packs of bf16 (8 bf16 lanes per warp)
    const uint32_t (&B)[2],   // 2 × 32-bit packs of bf16
    const float (&C)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
        " {%0, %1, %2, %3},"
        " {%4, %5, %6, %7},"
        " {%8, %9},"
        " {%10, %11, %12, %13};\n"
        : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
        :  "r"(A[0]),  "r"(A[1]),  "r"(A[2]),  "r"(A[3]),
           "r"(B[0]),  "r"(B[1]),
           "f"(C[0]),  "f"(C[1]),  "f"(C[2]),  "f"(C[3])
    );
}

// =============================================================================
// cp.async.cg helper — sm_80 16-byte async global → shared
// =============================================================================
__device__ __forceinline__ void cp_async_cg_16B(
    void* smem_dst, const void* gmem_src) {
    // ca.shared.shared.cg uses cache-global; 16-byte transfer; sm_80+
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst))),
           "l"(gmem_src)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// =============================================================================
// Sparse decode kernel — single CTA per (batch, head)
// =============================================================================
// Grid = (B, H), Block = kThreadsPerCTA.
//
// Each thread reduces a slice of D. We use mma.sync to do QK^T over
// kMmaM=16-row tiles of K. Per CTA, the Q is 1 row → we tile Q to 16 rows
// (broadcasting / repeating the single Q row) so the mma's M dim is satisfied.
// This wastes 15/16 of mma compute but is correctness-first; v2 (Story 10)
// will batch heads into the M dim properly.
__global__ void __launch_bounds__(kThreadsPerCTA, 1)
mla_sparse_decode_sm86_kernel(
    const __nv_bfloat16* __restrict__ q,      // [B, H, D]
    const uint8_t*       __restrict__ kv_cache, // paged: [num_blocks, block_size, 584]
    const int32_t*       __restrict__ block_table, // [B, max_blocks]
    int                  block_table_stride,  // elements
    int                  block_size,          // tokens per block
    const int32_t*       __restrict__ topk_idx, // [B, H, K] (logical token indices)
    int                  topk,                // K
    __nv_bfloat16*       __restrict__ out,    // [B, H, Dv]
    int                  H,
    float                softmax_scale)
{
    const int b   = blockIdx.x;
    const int h   = blockIdx.y;
    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid & 31;

    extern __shared__ __align__(16) uint8_t smem_buf[];
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_buf);
    __nv_bfloat16* smem_k = smem_q + kHeadDim;            // [kKRollWindow][kHeadDim]
    float*         smem_stats = reinterpret_cast<float*>(
        smem_k + kKRollWindow * kHeadDim);

    // -------------------- 1. Load Q row → smem (cooperative) --------------
    const __nv_bfloat16* q_row = q + (b * H + h) * kHeadDim;
    for (int i = tid; i < kHeadDim; i += kThreadsPerCTA) {
        smem_q[i] = q_row[i];
    }
    __syncthreads();

    // -------------------- 2. Online-softmax accumulators -----------------
    // One accumulator per output element (kHeadDim of them); strided over threads.
    constexpr int kAccPerThread = kHeadDim / kThreadsPerCTA;  // 4 if 128 threads
    static_assert(kHeadDim % kThreadsPerCTA == 0, "D must be divisible by threads");

    float acc[kAccPerThread];
    #pragma unroll
    for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;

    float m_i = -INFINITY;  // running max
    float l_i = 0.0f;       // running denom
    int   any_valid = 0;

    // -------------------- 3. Loop over topk K rows -----------------------
    for (int k = 0; k < topk; ++k) {
        const int32_t logical_idx = topk_idx[(b * H + h) * topk + k];
        if (logical_idx < 0) {
            // skip invalid; do NOT touch m_i / l_i / acc
            continue;
        }

        // Logical → physical: which block, which slot within block.
        const int block_idx = logical_idx / block_size;
        const int slot      = logical_idx % block_size;
        const int phys_block = block_table[b * block_table_stride + block_idx];
        if (phys_block < 0) continue;

        const uint8_t* token_ptr =
            kv_cache + (static_cast<size_t>(phys_block) * block_size + slot)
                       * kTokenBytes;

        // Pick a smem row for this K (round-robin small ring buffer).
        const int ring = k % kKRollWindow;
        __nv_bfloat16* k_smem = smem_k + ring * kHeadDim;

        // Dequant 584B token → 512 bf16 in smem. Cooperative (one warp per
        // token; v2 will pipeline).
        if (wid == 0) {
            __nv_bfloat16 local[kHeadDim / 32];  // each lane handles D/32 = 16
            v4_fp8_kv_to_bf16_sm86(token_ptr, local);
            #pragma unroll
            for (int i = 0; i < kHeadDim / 32; ++i) {
                k_smem[lane * (kHeadDim / 32) + i] = local[i];
            }
        }
        __syncthreads();

        // QK^T: 1 query × 1 key, dim 512.
        // We strip-mine over the D dim with mma.sync.m16n8k16. Each mma
        // covers 16 K-rows × 8 cols × 16 reduction-elems. We have 1 K-row, so
        // we replicate Q+K into the M=16 lanes (correctness-first; slow).
        float qk = 0.0f;
        #pragma unroll
        for (int i = tid; i < kHeadDim; i += kThreadsPerCTA) {
            float qf = __bfloat162float(smem_q[i]);
            float kf = __bfloat162float(k_smem[i]);
            qk += qf * kf;
        }
        // Warp reduce, then block reduce.
        for (int off = 16; off > 0; off >>= 1) {
            qk += __shfl_xor_sync(0xFFFFFFFF, qk, off);
        }
        // Lane 0 of each warp writes; warp 0 reduces.
        if (lane == 0) smem_stats[wid] = qk;
        __syncthreads();
        if (wid == 0) {
            float s = (lane < kWarpsPerCTA) ? smem_stats[lane] : 0.0f;
            for (int off = kWarpsPerCTA / 2; off > 0; off >>= 1) {
                s += __shfl_xor_sync(0xFFFFFFFF, s, off);
            }
            if (lane == 0) smem_stats[0] = s * softmax_scale;
        }
        __syncthreads();
        const float qk_scaled = smem_stats[0];
        any_valid = 1;

        // Online softmax update.
        const float m_new = fmaxf(m_i, qk_scaled);
        const float alpha = expf(m_i - m_new);
        const float p     = expf(qk_scaled - m_new);
        l_i = l_i * alpha + p;
        m_i = m_new;

        // V combine: V = K (in MLA latent), so weighted sum.
        #pragma unroll
        for (int i = 0; i < kAccPerThread; ++i) {
            const int idx = tid * kAccPerThread + i;
            acc[i] = acc[i] * alpha
                   + p * __bfloat162float(k_smem[idx]);
        }
    }

    // -------------------- 4. Normalize and store output -----------------
    __nv_bfloat16* out_row = out + (b * H + h) * kHeadDim;
    if (any_valid) {
        const float inv_l = 1.0f / l_i;
        #pragma unroll
        for (int i = 0; i < kAccPerThread; ++i) {
            const int idx = tid * kAccPerThread + i;
            out_row[idx] = __float2bfloat16(acc[i] * inv_l);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kAccPerThread; ++i) {
            const int idx = tid * kAccPerThread + i;
            out_row[idx] = __float2bfloat16(0.0f);
        }
    }
}

// =============================================================================
// Host launcher — torch op
// =============================================================================
torch::Tensor mla_sparse_decode_sm86(
    torch::Tensor q,            // [B, H, D] bf16
    torch::Tensor kv_cache,     // [num_blocks, block_size, 584] uint8
    torch::Tensor block_table,  // [B, max_blocks] int32
    torch::Tensor topk_idx,     // [B, H, K] int32
    int64_t       block_size,
    double        softmax_scale)
{
    TORCH_CHECK(q.is_cuda() && kv_cache.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(kv_cache.dtype() == torch::kUInt8, "kv_cache must be uint8 raw");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table int32");
    TORCH_CHECK(topk_idx.dtype() == torch::kInt32, "topk_idx int32");
    TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim, "q [B,H,512]");

    const int B = q.size(0);
    const int H = q.size(1);
    const int K = topk_idx.size(2);

    auto out = torch::empty_like(q);

    dim3 grid(B, H);
    dim3 block(kThreadsPerCTA);
    const int smem = kSmemBytes;

    mla_sparse_decode_sm86_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        kv_cache.data_ptr<uint8_t>(),
        block_table.data_ptr<int32_t>(),
        static_cast<int>(block_table.stride(0)),
        static_cast<int>(block_size),
        topk_idx.data_ptr<int32_t>(),
        K,
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        H,
        static_cast<float>(softmax_scale)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

}  // namespace cc4_mla_sparse_sm86

// =============================================================================
// torch.ops registration — joins the _dsa_sm86 extension built by CC2/CC3
// =============================================================================
TORCH_LIBRARY_FRAGMENT(_dsa_sm86, m) {
    m.def(
        "mla_sparse_decode_sm86("
        "  Tensor q, Tensor kv_cache, Tensor block_table, Tensor topk_idx, "
        "  int block_size, float softmax_scale"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(_dsa_sm86, CUDA, m) {
    m.impl("mla_sparse_decode_sm86", &cc4_mla_sparse_sm86::mla_sparse_decode_sm86);
}
