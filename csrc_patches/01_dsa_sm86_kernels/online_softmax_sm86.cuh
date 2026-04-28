// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// online_softmax_sm86.cuh — streaming online-softmax helper for sm_86 MLA-decode.
//
// Algorithm-preserving port of the inner-loop softmax used by FlashMLA's
// sparse decode (and CC4's Triton mla_decode_sparse_sm86). Pure header, all
// device-inline; include from sparse_attn_indexer_sm86.cu's downstream
// MLA-decode kernel (Story 6: "Online softmax + indexer top-k — algorithm-
// preserving, just data-movement substituted").
//
// Numerics: fp32 accumulator throughout (m_i max, l_i normalization sum, acc
// weighted V combine). bf16 inputs from cp.async.cg.shared.global tile loads.
//
// Usage pattern (single-warp accumulator, decode T_q=1):
//
//   OnlineSoftmaxState st;
//   st.reset();
//   for (int i = 0; i < num_keys; ++i) {
//     float qk = compute_qk_dot(...);              // mma.sync.aligned.m16n8k16
//     float v_row = load_v_row(...);                // cp.async.cg
//     st.update(qk, v_row);
//   }
//   float out = st.finalize();
//
// Edge case: zero-valid-keys (all topk == -1) — finalize() returns 0.
// Matches CC4's mla_decode_sparse_sm86 early-exit zero-output behavior.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace vllm {
namespace dsa_sm86 {

// ============================================================================
// Single-element online softmax accumulator (per-output-element scope)
// ============================================================================

struct OnlineSoftmaxScalar {
  float m_i;     // running max
  float l_i;     // running sum of exp(scores - m_i)
  float acc;     // running weighted V sum
  int   n_valid; // count of valid keys (== 0 -> finalize returns 0)

  __device__ __forceinline__ void reset() {
    m_i     = -FLT_MAX;
    l_i     = 0.0f;
    acc     = 0.0f;
    n_valid = 0;
  }

  // Update with one (qk score, v value) pair. `valid` masks out -1 indices
  // (sparse path: skip pairs where the topk slot was invalid).
  __device__ __forceinline__ void update(float qk, float v, bool valid) {
    if (!valid) return;
    const float qk_safe = qk;
    const float m_new   = fmaxf(m_i, qk_safe);
    const float alpha   = expf(m_i - m_new);
    const float p       = expf(qk_safe - m_new);
    acc = acc * alpha + p * v;
    l_i = l_i * alpha + p;
    m_i = m_new;
    ++n_valid;
  }

  __device__ __forceinline__ float finalize() const {
    if (n_valid == 0) return 0.0f;
    return acc / l_i;
  }
};

// ============================================================================
// Vector variant — accumulates a [VEC] block of V dimensions per element.
// One thread owns VEC components of the V-row across iterations.
// ============================================================================

template <int VEC>
struct OnlineSoftmaxVec {
  float m_i;
  float l_i;
  float acc[VEC];
  int   n_valid;

  __device__ __forceinline__ void reset() {
    m_i     = -FLT_MAX;
    l_i     = 0.0f;
    n_valid = 0;
#pragma unroll
    for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;
  }

  // `v_vec` holds VEC components of the K-th V row; `valid` is the topk mask.
  __device__ __forceinline__ void update(
      float qk,
      const float (&v_vec)[VEC],
      bool valid) {
    if (!valid) return;
    const float m_new = fmaxf(m_i, qk);
    const float alpha = expf(m_i - m_new);
    const float p     = expf(qk - m_new);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] = acc[i] * alpha + p * v_vec[i];
    }
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

// ============================================================================
// Warp-level reduction (combine per-thread online states across the warp).
// Used when the K dimension is partitioned across the 32 lanes of a warp.
// ============================================================================
//
// Each lane holds a partial OnlineSoftmaxScalar over a subset of K rows.
// We need to combine them into a single state on lane 0. This is the
// numerically-stable two-pass reduction:
//   m_combined = max(m_a, m_b)
//   l_combined = l_a * exp(m_a - m_combined) + l_b * exp(m_b - m_combined)
//   acc_combined = acc_a * exp(m_a - m_combined) + acc_b * exp(m_b - m_combined)

__device__ __forceinline__ OnlineSoftmaxScalar warp_reduce_online_softmax(
    OnlineSoftmaxScalar st) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float m_other       = __shfl_down_sync(0xFFFFFFFF, st.m_i, offset);
    float l_other       = __shfl_down_sync(0xFFFFFFFF, st.l_i, offset);
    float acc_other     = __shfl_down_sync(0xFFFFFFFF, st.acc, offset);
    int   n_other       = __shfl_down_sync(0xFFFFFFFF, st.n_valid, offset);

    if (n_other > 0) {
      const float m_new = fmaxf(st.m_i, m_other);
      const float alpha_self  = (st.n_valid > 0) ? expf(st.m_i - m_new) : 0.0f;
      const float alpha_other = expf(m_other - m_new);
      st.l_i  = st.l_i  * alpha_self + l_other  * alpha_other;
      st.acc  = st.acc  * alpha_self + acc_other * alpha_other;
      st.m_i  = m_new;
      st.n_valid += n_other;
    }
  }
  return st;
}

}  // namespace dsa_sm86
}  // namespace vllm

// // --ProtoAI-Bakari--
