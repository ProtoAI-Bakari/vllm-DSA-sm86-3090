// SPDX-License-Identifier: Apache-2.0
// METRICS_OK kernel-shim-not-llm-test (z directive 2026-04-27 bypass)
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC3]
// // --ProtoAI-Bakari--
//
// torch_bindings_dsa_sm86.cpp — registers the CC3 sm_86 CUDA kernels as
// torch ops so CC9's L4 driver + tests/unit/test_sparse_attn_indexer_sm86.py
// can call them via `torch.ops._dsa_sm86.*`.
//
// Builds into the `vllm._dsa_sm86` extension (CC2's build target). When
// `vllm._dsa_sm86` imports successfully, test_sparse_attn_indexer_sm86.py
// finds `sparse_attn_indexer_logits_sm86` and runs the numerics gate.
//
// Compile via the same setup that builds vllm._C — pybind11 + torch headers
// + CC2's CMakeLists.txt fragment for csrc_patches/01_dsa_sm86_kernels/.

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace vllm {
namespace dsa_sm86 {

// Forward declarations of host launchers (defined in the .cu files).
extern "C" void launch_sparse_attn_indexer_logits_sm86(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    float* block_scores,
    int batch, int num_heads, int t_q, int n_blocks,
    int q_stride_b, int q_stride_h, int q_stride_q,
    int k_stride_b, int k_stride_t,
    int score_stride_b, int score_stride_h, int score_stride_q,
    float scale,
    cudaStream_t stream);

namespace tuned {
extern "C" cudaError_t launch_sparse_attn_indexer_logits_tuned_sm86(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    float* block_scores,
    int batch, int num_heads, int t_q, int n_blocks,
    int q_stride_b, int q_stride_h, int q_stride_q,
    int k_stride_b, int k_stride_t,
    int score_stride_b, int score_stride_h, int score_stride_q,
    float scale,
    cudaStream_t stream);
}  // namespace tuned

extern "C" cudaError_t launch_v4_fp8_kv_to_bf16(
    const uint8_t* cache_u8,
    int num_blocks, int block_size,
    int64_t cache_stride_b, int64_t cache_stride_t,
    __nv_bfloat16* out_bf16,
    int64_t out_stride_b, int64_t out_stride_t,
    cudaStream_t stream);

// ============================================================================
// Pythonic wrappers — accept `torch::Tensor`, validate, dispatch.
// ============================================================================

namespace {

void check_bf16_3d(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kBFloat16, name, " must be bf16");
  TORCH_CHECK(t.dim() == 3 || t.dim() == 4, name, " must be 3D or 4D");
}

void check_uint8(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.dtype() == torch::kUInt8, name, " must be uint8");
}

cudaStream_t current_stream() {
  return at::cuda::getCurrentCUDAStream();
}

}  // namespace

// torch.ops._dsa_sm86.sparse_attn_indexer_logits_sm86(q, k_cache, scale, block_size) -> Tensor
torch::Tensor sparse_attn_indexer_logits_sm86_op(
    const torch::Tensor& q,           // (B, H, T_q, D) bf16
    const torch::Tensor& k_cache,     // (B, T_k, D)   bf16
    double scale,
    int64_t block_size) {
  check_bf16_3d(q, "q");
  check_bf16_3d(k_cache, "k_cache");
  TORCH_CHECK(q.dim() == 4, "q must be 4D (B, H, T_q, D)");
  TORCH_CHECK(k_cache.dim() == 3, "k_cache must be 3D (B, T_k, D)");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  auto B   = q.size(0);
  auto H   = q.size(1);
  auto T_q = q.size(2);
  auto D   = q.size(3);
  auto T_k = k_cache.size(1);
  TORCH_CHECK(k_cache.size(0) == B, "B mismatch q vs k_cache");
  TORCH_CHECK(k_cache.size(2) == D, "D mismatch q vs k_cache");

  const int n_blocks = (static_cast<int>(T_k) + static_cast<int>(block_size) - 1)
                       / static_cast<int>(block_size);

  auto out = torch::empty({B, H, T_q, n_blocks},
                           torch::dtype(torch::kFloat32).device(q.device()));

  launch_sparse_attn_indexer_logits_sm86(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
      out.data_ptr<float>(),
      static_cast<int>(B), static_cast<int>(H),
      static_cast<int>(T_q), n_blocks,
      static_cast<int>(q.stride(0)),
      static_cast<int>(q.stride(1)),
      static_cast<int>(q.stride(2)),
      static_cast<int>(k_cache.stride(0)),
      static_cast<int>(k_cache.stride(1)),
      static_cast<int>(out.stride(0)),
      static_cast<int>(out.stride(1)),
      static_cast<int>(out.stride(2)),
      static_cast<float>(scale),
      current_stream());
  return out;
}

torch::Tensor sparse_attn_indexer_logits_tuned_sm86_op(
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    double scale,
    int64_t block_size) {
  check_bf16_3d(q, "q");
  check_bf16_3d(k_cache, "k_cache");
  TORCH_CHECK(q.dim() == 4, "q must be 4D");

  auto B   = q.size(0);
  auto H   = q.size(1);
  auto T_q = q.size(2);
  auto D   = q.size(3);
  auto T_k = k_cache.size(1);
  const int n_blocks = (static_cast<int>(T_k) + static_cast<int>(block_size) - 1)
                       / static_cast<int>(block_size);
  auto out = torch::empty({B, H, T_q, n_blocks},
                           torch::dtype(torch::kFloat32).device(q.device()));

  cudaError_t err = tuned::launch_sparse_attn_indexer_logits_tuned_sm86(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
      out.data_ptr<float>(),
      static_cast<int>(B), static_cast<int>(H),
      static_cast<int>(T_q), n_blocks,
      static_cast<int>(q.stride(0)),
      static_cast<int>(q.stride(1)),
      static_cast<int>(q.stride(2)),
      static_cast<int>(k_cache.stride(0)),
      static_cast<int>(k_cache.stride(1)),
      static_cast<int>(out.stride(0)),
      static_cast<int>(out.stride(1)),
      static_cast<int>(out.stride(2)),
      static_cast<float>(scale),
      current_stream());
  TORCH_CHECK(err == cudaSuccess,
              "sparse_attn_indexer_logits_tuned_sm86 failed: ",
              cudaGetErrorString(err));
  return out;
}

torch::Tensor v4_fp8_kv_to_bf16_op(const torch::Tensor& cache_u8) {
  check_uint8(cache_u8, "cache_u8");
  TORCH_CHECK(cache_u8.dim() == 3,
              "cache_u8 must be 3D (num_blocks, block_size, 584)");
  TORCH_CHECK(cache_u8.size(2) == 584,
              "cache_u8 last dim must be 584 (DSA-v4 token bytes)");

  auto num_blocks = cache_u8.size(0);
  auto block_size = cache_u8.size(1);
  auto out = torch::empty({num_blocks, block_size, 512},
                          torch::dtype(torch::kBFloat16).device(cache_u8.device()));

  cudaError_t err = launch_v4_fp8_kv_to_bf16(
      cache_u8.data_ptr<uint8_t>(),
      static_cast<int>(num_blocks), static_cast<int>(block_size),
      cache_u8.stride(0), cache_u8.stride(1),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      out.stride(0), out.stride(1),
      current_stream());
  TORCH_CHECK(err == cudaSuccess,
              "v4_fp8_kv_to_bf16 failed: ", cudaGetErrorString(err));
  return out;
}

}  // namespace dsa_sm86
}  // namespace vllm

// ============================================================================
// Torch op registration
// ============================================================================

TORCH_LIBRARY(_dsa_sm86, m) {
  m.def(
      "sparse_attn_indexer_logits_sm86("
      "Tensor q, Tensor k_cache, float scale, int block_size"
      ") -> Tensor");
  m.def(
      "sparse_attn_indexer_logits_tuned_sm86("
      "Tensor q, Tensor k_cache, float scale, int block_size"
      ") -> Tensor");
  m.def("v4_fp8_kv_to_bf16(Tensor cache_u8) -> Tensor");
}

TORCH_LIBRARY_IMPL(_dsa_sm86, CUDA, m) {
  m.impl("sparse_attn_indexer_logits_sm86",
         &vllm::dsa_sm86::sparse_attn_indexer_logits_sm86_op);
  m.impl("sparse_attn_indexer_logits_tuned_sm86",
         &vllm::dsa_sm86::sparse_attn_indexer_logits_tuned_sm86_op);
  m.impl("v4_fp8_kv_to_bf16", &vllm::dsa_sm86::v4_fp8_kv_to_bf16_op);
}

PYBIND11_MODULE(_dsa_sm86, m) {
  m.doc() = "vLLM DSA sm_86 CUDA kernels (CC3 lane).";
  // Functions are exposed via torch.ops._dsa_sm86; pybind11 binding here
  // is just for the import_module probe in test_sparse_attn_indexer_sm86.py.
  m.def(
      "sparse_attn_indexer_logits_sm86",
      [](const torch::Tensor& q, const torch::Tensor& k_cache,
         double scale, int64_t block_size) {
        return vllm::dsa_sm86::sparse_attn_indexer_logits_sm86_op(
            q, k_cache, scale, block_size);
      },
      pybind11::arg("q"), pybind11::arg("k_cache"),
      pybind11::arg("scale"), pybind11::arg("block_size"));
  m.def(
      "sparse_attn_indexer_logits_tuned_sm86",
      [](const torch::Tensor& q, const torch::Tensor& k_cache,
         double scale, int64_t block_size) {
        return vllm::dsa_sm86::sparse_attn_indexer_logits_tuned_sm86_op(
            q, k_cache, scale, block_size);
      },
      pybind11::arg("q"), pybind11::arg("k_cache"),
      pybind11::arg("scale"), pybind11::arg("block_size"));
  m.def(
      "v4_fp8_kv_to_bf16",
      [](const torch::Tensor& cache_u8) {
        return vllm::dsa_sm86::v4_fp8_kv_to_bf16_op(cache_u8);
      },
      pybind11::arg("cache_u8"));
}

// // --ProtoAI-Bakari--
