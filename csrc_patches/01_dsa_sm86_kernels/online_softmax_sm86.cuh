// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// online_softmax_sm86.cuh — streaming online-softmax helper for sm_86 MLA-decode.
// Algorithm-preserving port of the FlashMLA sparse-decode inner-loop softmax.
// Pure header. Used by the MLA-decode .cu downstream of sparse_attn_indexer_sm86.cu.
//
// fp32 accumulator throughout. Empty-keys safe (returns 0, not NaN).

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace vllm {
namespace dsa_sm86 {

struct OnlineSoftmaxScalar {
  float m_i;
  float l_i;
  float acc;
  int   n_valid;

  __device__ __forceinline__ void reset() {
    m_i = -FLT_MAX; l_i = 0.0f; acc = 0.0f; n_valid = 0;
  }

  __device__ __forceinline__ void update(float qk, float v, bool valid) {
    if (!valid) return;
    const float m_new = fmaxf(m_i, qk);
    const float alpha = expf(m_i - m_new);
    const float p     = expf(qk - m_new);
    acc = acc * alpha + p * v;
    l_i = l_i * alpha + p;
    m_i = m_new;
    ++n_valid;
  }

  __device__ __forceinline__ float finalize() const {
    return n_valid == 0 ? 0.0f : acc / l_i;
  }
};

template <int VEC>
struct OnlineSoftmaxVec {
  float m_i;
  float l_i;
  float acc[VEC];
  int   n_valid;

  __device__ __forceinline__ void reset() {
    m_i = -FLT_MAX; l_i = 0.0f; n_valid = 0;
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;
  }

  __device__ __forceinline__ void update(
      float qk, const float (&v_vec)[VEC], bool valid) {
    if (!valid) return;
    const float m_new = fmaxf(m_i, qk);
    const float alpha = expf(m_i - m_new);
    const float p     = expf(qk - m_new);
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = acc[i] * alpha + p * v_vec[i];
    l_i = l_i * alpha + p;
    m_i = m_new;
    ++n_valid;
  }

  __device__ __forceinline__ void finalize(float (&out)[VEC]) const {
    if (n_valid == 0) {
#pragma unroll
      for (int i = 0; i < VEC; ++i) out[i] = 0.0f;
      return;
    }
    const float inv_l = 1.0f / l_i;
#pragma unroll
    for (int i = 0; i < VEC; ++i) out[i] = acc[i] * inv_l;
  }
};

// Numerically-stable warp reduction (combine partial states across 32 lanes).
__device__ __forceinline__ OnlineSoftmaxScalar warp_reduce_online_softmax(
    OnlineSoftmaxScalar st) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float m_other   = __shfl_down_sync(0xFFFFFFFF, st.m_i, offset);
    float l_other   = __shfl_down_sync(0xFFFFFFFF, st.l_i, offset);
    float acc_other = __shfl_down_sync(0xFFFFFFFF, st.acc, offset);
    int   n_other   = __shfl_down_sync(0xFFFFFFFF, st.n_valid, offset);
    if (n_other > 0) {
      const float m_new = fmaxf(st.m_i, m_other);
      const float a_self  = (st.n_valid > 0) ? expf(st.m_i - m_new) : 0.0f;
      const float a_other = expf(m_other - m_new);
      st.l_i  = st.l_i * a_self + l_other  * a_other;
      st.acc  = st.acc * a_self + acc_other * a_other;
      st.m_i  = m_new;
      st.n_valid += n_other;
    }
  }
  return st;
}

}  // namespace dsa_sm86
}  // namespace vllm

// // --ProtoAI-Bakari--
