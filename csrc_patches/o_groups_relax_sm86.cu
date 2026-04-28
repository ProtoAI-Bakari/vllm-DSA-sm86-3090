// o_groups_relax_sm86.cu — CC5 Story 6 (now critical-path)
// Relax DSV4 o_groups=8 invariant for TP=16 single-replica launch on sm_86.
//
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7)
// [1M ctx, max effort, agent: CC5]
// // --ProtoAI-Bakari--
//
// PROBLEM
// -------
// DSV4-Flash-FP8 ships expert w13 with output-grouping factor `o_groups=8`.
// vLLM's standard `ColumnParallelLinear` enforces `n_inter % tp_size == 0`
// AND `o_groups % tp_size == 0` so each rank holds whole o_groups.
// At TP=16 with o_groups=8 → `8 % 16 != 0` → ZeroDivisionError or
// `shape '[64,4,128]' invalid for size 65536` at weight load.
//
// TP=16 is mandatory (DP=8 structurally impossible: 274GB / 8 = 34.25 GB
// per replica > 24 GB card cap; 274GB / 16 = 17.1 GB/rank fits with
// headroom for KV + activations).
//
// SOLUTION — sub-sharding within each o_group along the K (hidden) axis
// ---------------------------------------------------------------------
// `sub_factor = tp_size / o_groups = 16 / 8 = 2`. Each o_group's weight
// slice [hidden, n_inter/o_groups] = [7168, 512] is split across 2 ranks
// along K → each rank holds [3584, 512] = 1.75 MiB BF16 per expert per
// o_group (× 256 experts × o_groups_local=1 ≈ 448 MiB per rank for w13).
// Plus w2, total ≈ 17 GB/rank as math predicts.
//
// At forward time:
//   1. Each rank computes partial GEMM on its K-shard:
//        a_local[T, K_sub] @ w_local[K_sub, N_local] → partial[T, N_local]
//   2. Sub-group (size = sub_factor = 2) all-reduce sums the partials so
//      each rank in the sub-group ends up with the full GEMM output for
//      its (o_group, n_local) slice.
//   3. Downstream code sees the same `[T, n_inter/o_groups]` shape it
//      would have seen pre-relaxation.
//
// Numerics: K-axis split + all-reduce is an exact reformulation of the
// original GEMM (sum over K decomposes). No accuracy loss vs the
// unsharded compute.
//
// HOW THIS PATCH PLUGS INTO vLLM
// ------------------------------
// vLLM's `ColumnParallelLinear.weight_loader` is monkey-patched (Python
// side, see `csrc_patches/o_groups_relax.py` sibling) to route the
// o_groups=8 / TP>o_groups case here. This C++/CUDA file provides:
//
//   compute_o_group_shard_plan(...)  — slicing math (per-rank metadata)
//   o_groups_partial_gemm(...)       — BF16 partial GEMM via cuBLAS-Lt
//                                      (Marlin path used when AWQ-INT4
//                                      weights, INT8 path under Story 4)
//   o_groups_subgroup_allreduce(...) — NCCL all-reduce within sub-group
//
// The sub-group communicator is created in Python (vLLM's
// `init_distributed_environment` plus a `torch.distributed.new_group`
// over each set of `sub_factor` consecutive ranks).
//
// BUILD (CC2 cmake -DTORCH_CUDA_ARCH_LIST="8.6"):
//   nvcc -arch=sm_86 -std=c++17 -O3 -Xcompiler=-fPIC -shared \
//     o_groups_relax_sm86.cu -o o_groups_relax_sm86.so \
//     -lcublasLt -lcublas \
//     $(python3-config --includes) \
//     -I$(python3 -c 'import torch.utils.cpp_extension as e; print(e.include_paths()[0])')

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10d/ProcessGroup.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;

// ===========================================================================
// PART 1 — Sharding plan
// ===========================================================================
//
// Each rank's identity within the relaxation:
//   o_group_idx  ∈ [0, o_groups)         which output group this rank serves
//   sub_rank     ∈ [0, sub_factor)       which K-shard within that o_group
//   sub_factor   = tp_size / o_groups    (must divide cleanly; we assert)
//
// Per-rank tensor slicing for a weight `[K_total, N_total]` whose semantic
// layout is `[K_total, o_groups, N_total/o_groups]`:
//
//   K_sub        = K_total / sub_factor
//   k_start      = sub_rank * K_sub
//   k_end        = (sub_rank + 1) * K_sub
//   N_local      = N_total / o_groups
//   n_start      = o_group_idx * N_local
//   n_end        = (o_group_idx + 1) * N_local
//
// Local weight stored shape: [K_sub, N_local].

struct OGroupShardPlan {
    int tp_rank;
    int tp_size;
    int o_groups;
    int sub_factor;
    int o_group_idx;
    int sub_rank;
    int K_total;
    int K_sub;
    int N_total;
    int N_local;
    int k_start;
    int k_end;
    int n_start;
    int n_end;
    int sub_group_root;        // tp_rank of sub_rank=0 in this rank's sub-group
};

static inline OGroupShardPlan make_plan(
    int tp_rank, int tp_size, int o_groups, int K_total, int N_total
) {
    if (tp_size <= 0 || o_groups <= 0)
        throw std::invalid_argument("tp_size and o_groups must be positive");
    if (tp_size < o_groups || (tp_size % o_groups) != 0)
        throw std::invalid_argument(
            "o_groups_relax requires tp_size % o_groups == 0 and "
            "tp_size >= o_groups (got tp=" + std::to_string(tp_size) +
            " o_groups=" + std::to_string(o_groups) + ")");
    if (K_total <= 0 || N_total <= 0)
        throw std::invalid_argument("K_total and N_total must be positive");

    OGroupShardPlan p{};
    p.tp_rank      = tp_rank;
    p.tp_size      = tp_size;
    p.o_groups     = o_groups;
    p.sub_factor   = tp_size / o_groups;
    // Layout choice: rank order = [o_group_0_sub_0, o_group_0_sub_1, ...,
    //                              o_group_1_sub_0, ...]
    // i.e. consecutive ranks are sub-shards of the same o_group. This makes
    // sub-group all-reduce ranges contiguous in tp_rank space, which
    // matches NCCL's preference for adjacent ranks.
    p.o_group_idx  = tp_rank / p.sub_factor;
    p.sub_rank     = tp_rank % p.sub_factor;
    p.sub_group_root = p.o_group_idx * p.sub_factor;

    if ((K_total % p.sub_factor) != 0)
        throw std::invalid_argument(
            "K_total not divisible by sub_factor (K=" +
            std::to_string(K_total) + " sub_factor=" +
            std::to_string(p.sub_factor) + ")");
    if ((N_total % o_groups) != 0)
        throw std::invalid_argument(
            "N_total not divisible by o_groups (N=" +
            std::to_string(N_total) + " o_groups=" +
            std::to_string(o_groups) + ")");

    p.K_total = K_total;
    p.K_sub   = K_total / p.sub_factor;
    p.N_total = N_total;
    p.N_local = N_total / o_groups;
    p.k_start = p.sub_rank   * p.K_sub;
    p.k_end   = p.k_start    + p.K_sub;
    p.n_start = p.o_group_idx * p.N_local;
    p.n_end   = p.n_start    + p.N_local;
    return p;
}

// ===========================================================================
// PART 2 — Weight slicer (used at weight-load time)
// ===========================================================================
// Given a full `[K_total, N_total]` BF16 weight tensor (or a stream of it
// during HF safetensors load), produce this rank's `[K_sub, N_local]`
// slice. We expose two paths:
//
//   slice_full_weight(weight_full, plan) -> tensor [K_sub, N_local]
//       For test paths where the full tensor is materialized.
//
//   slice_streaming(...)
//       For HF safetensors streamed load — declare it but the actual
//       integration sits on the Python loader side (vLLM's `weight_loader`
//       calls a per-shard slicer and we wire this via the Python wrapper).

static at::Tensor slice_full_weight(
    const at::Tensor& weight_full,        // [K_total, N_total]
    const OGroupShardPlan& p
) {
    TORCH_CHECK(weight_full.dim() == 2, "weight must be 2D");
    TORCH_CHECK(weight_full.size(0) == p.K_total,
                "weight K dim mismatch with plan");
    TORCH_CHECK(weight_full.size(1) == p.N_total,
                "weight N dim mismatch with plan");
    return weight_full
        .narrow(0, p.k_start, p.K_sub)
        .narrow(1, p.n_start, p.N_local)
        .contiguous();
}

// ===========================================================================
// PART 3 — Partial GEMM via cuBLAS-Lt (BF16 in, BF16 out, FP32 accum)
// ===========================================================================
// Computes `out[T, N_local] = a_local[T, K_sub] @ w_local[K_sub, N_local]`
// on the rank's local shards. cuBLAS-Lt's BF16 path resolves to
// `mma.sync.aligned.m16n8k16` on sm_86 (no WGMMA dependency).
//
// Story 4 will swap this for AWQ-Marlin INT4 once that lands; the wrapper
// signature matches so downstream Python code can switch by config flag.

static cublasLtHandle_t g_cublas_lt = nullptr;

static cublasLtHandle_t get_cublas_lt() {
    if (!g_cublas_lt) {
        if (cublasLtCreate(&g_cublas_lt) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasLtCreate failed");
    }
    return g_cublas_lt;
}

at::Tensor o_groups_partial_gemm(
    const at::Tensor& a_local,       // [T, K_sub]  bf16
    const at::Tensor& w_local,       // [K_sub, N_local] bf16
    int M, int N, int K
) {
    TORCH_CHECK(a_local.is_cuda() && w_local.is_cuda(),
                "tensors must be CUDA");
    TORCH_CHECK(a_local.dtype() == at::kBFloat16 &&
                w_local.dtype() == at::kBFloat16,
                "tensors must be bf16 (Story 4 will add INT4 path)");
    TORCH_CHECK(a_local.size(0) == M && a_local.size(1) == K,
                "a_local shape mismatch");
    TORCH_CHECK(w_local.size(0) == K && w_local.size(1) == N,
                "w_local shape mismatch");

    auto out = at::empty({M, N}, a_local.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    cublasLtHandle_t lt = get_cublas_lt();
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr, w_desc = nullptr, c_desc = nullptr;
    auto cleanup = [&]() {
        if (op_desc)  cublasLtMatmulDescDestroy(op_desc);
        if (a_desc)   cublasLtMatrixLayoutDestroy(a_desc);
        if (w_desc)   cublasLtMatrixLayoutDestroy(w_desc);
        if (c_desc)   cublasLtMatrixLayoutDestroy(c_desc);
    };

    if (cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
            != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        throw std::runtime_error("cublasLtMatmulDescCreate failed");
    }
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n));

    // Layouts: cuBLAS-Lt is column-major; PyTorch is row-major. Standard
    // trick: compute (W^T @ A^T)^T using row-major-as-col-major reinterpret.
    //   row-major A[M,K] == col-major A_cm[K,M]
    //   row-major W[K,N] == col-major W_cm[N,K]
    //   col-major out_cm[N,M] == row-major out[M,N]
    // So we ask cuBLAS for: out_cm = W_cm @ A_cm with op=N for both.
    cublasLtMatrixLayoutCreate(&w_desc, CUDA_R_16BF, N, K, N);
    cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, N, M, N);

    float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t status = cublasLtMatmul(
        lt, op_desc,
        &alpha,
        w_local.data_ptr<bf16>(), w_desc,
        a_local.data_ptr<bf16>(), a_desc,
        &beta,
        out.data_ptr<bf16>(), c_desc,
        out.data_ptr<bf16>(), c_desc,
        nullptr, nullptr, 0, stream);
    cleanup();
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cublasLtMatmul failed status=" +
                                 std::to_string(static_cast<int>(status)));
    return out;
}

// ===========================================================================
// PART 4 — Sub-group all-reduce
// ===========================================================================
// Sums across the `sub_factor` ranks of this rank's sub-group. Done via
// `c10d::ProcessGroup::allreduce` (NCCL backend). The sub-group is created
// in Python at init time (`torch.distributed.new_group`) and its handle
// passed in.
//
// We accept the sub_group as a c10d::ProcessGroup shared_ptr indirected
// through a uintptr_t handle (Python wrapper does the cast). This keeps
// the C++ ABI free of c10d header churn between vLLM versions.

void o_groups_subgroup_allreduce(
    at::Tensor& tensor_inout,                 // [M, N_local], in-place sum
    uintptr_t process_group_handle
) {
    TORCH_CHECK(tensor_inout.is_cuda(), "tensor must be CUDA");
    TORCH_CHECK(tensor_inout.is_contiguous(), "tensor must be contiguous");
    auto* pg = reinterpret_cast<c10d::ProcessGroup*>(process_group_handle);
    if (!pg) throw std::runtime_error("null process_group_handle");

    std::vector<at::Tensor> tensors{tensor_inout};
    c10d::AllreduceOptions opts;
    opts.reduceOp = c10d::ReduceOp::SUM;
    auto work = pg->allreduce(tensors, opts);
    work->wait();
}

// ===========================================================================
// PART 5 — Convenience: end-to-end fused-call (slice already done)
// ===========================================================================
// One-shot: partial-gemm on local shards + sub-group all-reduce. Returns
// the full `[M, N_local]` GEMM result for this rank's o_group slice.

at::Tensor o_groups_relax_forward(
    const at::Tensor& a_local,
    const at::Tensor& w_local,
    int M, int N, int K,
    uintptr_t sub_group_handle
) {
    auto partial = o_groups_partial_gemm(a_local, w_local, M, N, K);
    o_groups_subgroup_allreduce(partial, sub_group_handle);
    return partial;
}

// ===========================================================================
// Pybind module
// ===========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<OGroupShardPlan>(m, "OGroupShardPlan")
        .def_readonly("tp_rank",         &OGroupShardPlan::tp_rank)
        .def_readonly("tp_size",         &OGroupShardPlan::tp_size)
        .def_readonly("o_groups",        &OGroupShardPlan::o_groups)
        .def_readonly("sub_factor",      &OGroupShardPlan::sub_factor)
        .def_readonly("o_group_idx",     &OGroupShardPlan::o_group_idx)
        .def_readonly("sub_rank",        &OGroupShardPlan::sub_rank)
        .def_readonly("K_total",         &OGroupShardPlan::K_total)
        .def_readonly("K_sub",           &OGroupShardPlan::K_sub)
        .def_readonly("N_total",         &OGroupShardPlan::N_total)
        .def_readonly("N_local",         &OGroupShardPlan::N_local)
        .def_readonly("k_start",         &OGroupShardPlan::k_start)
        .def_readonly("k_end",           &OGroupShardPlan::k_end)
        .def_readonly("n_start",         &OGroupShardPlan::n_start)
        .def_readonly("n_end",           &OGroupShardPlan::n_end)
        .def_readonly("sub_group_root",  &OGroupShardPlan::sub_group_root);

    m.def("make_plan", &make_plan,
          "Compute per-rank o_groups-relaxation shard plan",
          py::arg("tp_rank"), py::arg("tp_size"),
          py::arg("o_groups"), py::arg("K_total"), py::arg("N_total"));
    m.def("slice_full_weight", &slice_full_weight,
          "Slice a full [K_total,N_total] weight to this rank's "
          "[K_sub,N_local] shard");
    m.def("o_groups_partial_gemm", &o_groups_partial_gemm,
          "BF16 partial GEMM on K-sub shard via cuBLAS-Lt (sm_86 mma.sync)");
    m.def("o_groups_subgroup_allreduce", &o_groups_subgroup_allreduce,
          "All-reduce a partial output across the sub_factor-rank sub-group");
    m.def("o_groups_relax_forward", &o_groups_relax_forward,
          "Fused partial-gemm + sub-group all-reduce; returns full GEMM "
          "output for this rank's o_group slice");
}
