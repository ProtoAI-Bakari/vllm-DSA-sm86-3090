// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// fp8_kv_dequant_sm86.cu — DSA-v4 FP8 KV cache dequantizer for sm_86.
//
// Companion to sparse_attn_indexer_sm86.cu. The Lightning Indexer + MLA
// decode kernels expect bf16 K rows for the m16n8k16 mma path. The vLLM
// fp8_ds_mla cache stores DSA-v4 in 584-byte tokens:
//
//   [0..447]   = 448 × float8_e4m3 (NoPE, quantized in 7 groups × 64 elems)
//   [448..575] = 64  × bfloat16    (RoPE, unquantized)
//   [576..582] = 7   × ue8m0       (per-64-element NoPE block scales)
//   [583]      = 1   × pad
//
// This kernel reads (num_blocks, block_size, 584) uint8 view of the cache and
// writes (num_blocks, block_size, 512) bf16 (448 NoPE + 64 RoPE).
//
// Pure C++/CUDA (no Triton). Used by the host-side flash_mla shim before
// invoking the bf16 sparse_attn_indexer / mla_decode kernels. Stream-order
// implicit sync replaces Hopper PDL.
//
// Compile: -arch=sm_86 -std=c++17

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace vllm {
namespace dsa_sm86 {

// ============================================================================
// Layout constants — DSA-v4 fp8_ds_mla
// ============================================================================
constexpr int kV4NopeBytes   = 448;
constexpr int kV4RopeElems   = 64;
constexpr int kV4RopeBytes   = kV4RopeElems * 2;       // 128
constexpr int kV4ScaleBytes  = 7;
constexpr int kV4PadBytes    = 1;
constexpr int kV4TokenBytes  = kV4NopeBytes + kV4RopeBytes + kV4ScaleBytes + kV4PadBytes;
constexpr int kV4NopeGroup   = 64;
constexpr int kV4OutDim      = kV4NopeBytes + kV4RopeElems;   // 448 + 64 = 512

static_assert(kV4TokenBytes == 584, "DSA-v4 token stride must be 584 B");

// ============================================================================
// FP8 e4m3 -> fp32 decoder (bit-pattern, sm_80+ compatible)
// E4M3: sign(1) | exp(4, bias 7) | mantissa(3)
// Subnormals: (-1)^s * 2^(-6) * (mantissa/8)
// Normals:    (-1)^s * 2^(exp-7) * (1 + mantissa/8)
// Saturate / NaN / Inf: not emitted by the encoder; accept as raw decode.
// ============================================================================

__device__ __forceinline__ float fp8_e4m3_to_fp32(uint8_t b) {
  const uint32_t sign  = (b >> 7) & 1;
  const uint32_t exp   = (b >> 3) & 0xF;
  const uint32_t mant  = b & 0x7;

  float val;
  if (exp == 0) {
    if (mant == 0) {
      val = 0.0f;
    } else {
      // Subnormal: 2^-6 * (mant/8)
      val = ldexpf(static_cast<float>(mant) / 8.0f, -6);
    }
  } else {
    // Normal: 2^(exp-7) * (1 + mant/8)
    const float significand = 1.0f + static_cast<float>(mant) / 8.0f;
    val = ldexpf(significand, static_cast<int>(exp) - 7);
  }
  return sign ? -val : val;
}

// ============================================================================
// ue8m0 byte → fp32 scale: 2^(byte - 127)
// ============================================================================
__device__ __forceinline__ float ue8m0_to_fp32(uint8_t b) {
  return ldexpf(1.0f, static_cast<int>(b) - 127);
}

// ============================================================================
// Kernel — one program per (block_idx, token_idx). Each thread covers
// one element of the 512-wide output (448 NoPE + 64 RoPE).
// ============================================================================

__global__ void v4_fp8_kv_to_bf16_kernel(
    const uint8_t* __restrict__ cache_u8,    // (B, T, 584) uint8
    int64_t cache_stride_b,
    int64_t cache_stride_t,
    __nv_bfloat16* __restrict__ out_bf16,    // (B, T, 512) bf16
    int64_t out_stride_b,
    int64_t out_stride_t,
    int block_size) {
  const int blk = blockIdx.x;
  const int tok = blockIdx.y;
  const int tx  = threadIdx.x;

  const uint8_t* token_base = cache_u8 + blk * cache_stride_b + tok * cache_stride_t;
  __nv_bfloat16* out_base = out_bf16 + blk * out_stride_b + tok * out_stride_t;

  // ---- Decode the 7 ue8m0 scales into shared memory ----
  __shared__ float scales[8];  // pad to 8 for warp-aligned access
  if (tx < kV4ScaleBytes) {
    const uint8_t s_byte = token_base[kV4NopeBytes + kV4RopeBytes + tx];
    scales[tx] = ue8m0_to_fp32(s_byte);
  } else if (tx == kV4ScaleBytes) {
    scales[tx] = 1.0f;  // unused
  }
  __syncthreads();

  // ---- NoPE: 448 elements, 7 groups of 64. Each thread covers ≥1 element. ----
  // We launch with kThreadsPerBlock = 512 → 1 thread per output element exactly.
  if (tx < kV4NopeBytes) {
    const int group = tx / kV4NopeGroup;       // 0..6
    const float scale = scales[group];
    const float val = fp8_e4m3_to_fp32(token_base[tx]) * scale;
    out_base[tx] = __float2bfloat16(val);
  } else if (tx < kV4NopeBytes + kV4RopeElems) {
    // RoPE: 64 bf16 elements at byte offset 448. Already bf16 — copy.
    const int rope_idx = tx - kV4NopeBytes;  // 0..63
    const __nv_bfloat16* rope_in = reinterpret_cast<const __nv_bfloat16*>(
        token_base + kV4NopeBytes + rope_idx * 2);
    out_base[kV4NopeBytes + rope_idx] = *rope_in;
  }
}

// ============================================================================
// Host launcher
// ============================================================================

extern "C" cudaError_t launch_v4_fp8_kv_to_bf16(
    const uint8_t* cache_u8,
    int num_blocks, int block_size,
    int64_t cache_stride_b, int64_t cache_stride_t,
    __nv_bfloat16* out_bf16,
    int64_t out_stride_b, int64_t out_stride_t,
    cudaStream_t stream) {
  // 1 thread per output element (448 NoPE + 64 RoPE = 512).
  dim3 grid(num_blocks, block_size);
  dim3 block(kV4OutDim);   // 512 threads/CTA (within sm_86 1024-thread limit)

  v4_fp8_kv_to_bf16_kernel<<<grid, block, 0, stream>>>(
      cache_u8,
      cache_stride_b,
      cache_stride_t,
      out_bf16,
      out_stride_b,
      out_stride_t,
      block_size);
  return cudaGetLastError();
}

}  // namespace dsa_sm86
}  // namespace vllm

// // --ProtoAI-Bakari--
