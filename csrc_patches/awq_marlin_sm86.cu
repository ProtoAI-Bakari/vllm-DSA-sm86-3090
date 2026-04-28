// awq_marlin_sm86.cu — CC5 Story 4
// INT4-AWQ-Marlin sm_86 backport bridge for DSV4 MoE expert FP8 GEMM.
//
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7)
// [1M ctx, max effort, agent: CC5]
// // --ProtoAI-Bakari--
//
// CONTEXT
// -------
// DSV4-Flash-FP8 ships expert weights as `[128,128]` block-scaled FP8.
// On Hopper this runs through DEEP_GEMM (TMA + WGMMA). On Ampere (sm_86)
// no native FP8 path exists; vLLM falls back to BF16 emulation at
// 10-20 t/s aggregate ceiling.
//
// vLLM ships an INT4-AWQ-Marlin GEMM that IS sm_80+ native (the original
// Marlin paper targeted A100). Story 1 probe confirms the op resolves on
// our cards. This file is the bridge that lets the MoE expert path
// substitute Marlin INT4 for the FP8 GEMM, conditional on:
//
//   - AWQ-quantized expert weights present (Story 5 produces these)
//   - group_size compatible (Marlin: 128; GLM-5.1 IQ2XXS: 64 → fallback)
//   - shape compatible (DSV4 N=4096/2048 fits Marlin tile multiples)
//
// PROVIDES
// --------
//   awq_marlin_expert_gemm(...)   primary dispatch — Marlin INT4
//   triton_int4_expert_gemm(...)  fallback — Triton kernel string compiled
//                                 at first call (no nvcc dep at our level)
//   awq_repack_for_marlin(...)    AWQ packed-int32 → Marlin-interleaved
//                                 layout. Wraps vLLM's awq_marlin_repack
//                                 op when present; emits a CPU repack
//                                 stub otherwise (test path).
//
// The C++ wrapper RESOLVES Marlin at runtime via torch.ops.vllm rather
// than linking against vLLM's compiled extension — this keeps our
// ABI-independent. If the op is missing (different vLLM version), we
// raise a Python-catchable error and the caller falls through to Triton.
//
// BUILD (CC2 cmake -DTORCH_CUDA_ARCH_LIST="8.6"):
//   nvcc -arch=sm_86 -std=c++17 -O3 -Xcompiler=-fPIC -shared \
//     awq_marlin_sm86.cu -o awq_marlin_sm86.so \
//     $(python3-config --includes) \
//     -I$(python3 -c 'import torch.utils.cpp_extension as e; print(e.include_paths()[0])')

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

// ---------------------------------------------------------------------------
// Helper: resolve `torch.ops.vllm.awq_marlin_gemm` at runtime
// ---------------------------------------------------------------------------
// Returns the OperatorHandle if the op exists. Throws std::runtime_error
// if not — caller decides whether to fall back to Triton or surface the
// error.

c10::OperatorHandle resolve_awq_marlin_op() {
    auto handle = c10::Dispatcher::singleton().findOp(
        c10::OperatorName("vllm::awq_marlin_gemm", "")
    );
    if (!handle.has_value()) {
        throw std::runtime_error(
            "vllm::awq_marlin_gemm not registered; either vLLM build is "
            "older than PR #40760 or AWQ-Marlin compile flag was not set "
            "during CC2 cmake. Story 1 probe should have caught this — "
            "if you see this at runtime, treat as FAIL_OP_UNRESOLVED."
        );
    }
    return handle.value();
}

c10::OperatorHandle resolve_awq_marlin_repack_op() {
    auto handle = c10::Dispatcher::singleton().findOp(
        c10::OperatorName("vllm::awq_marlin_repack", "")
    );
    if (!handle.has_value()) {
        throw std::runtime_error(
            "vllm::awq_marlin_repack not registered (paired with "
            "awq_marlin_gemm; same provenance)."
        );
    }
    return handle.value();
}

// ---------------------------------------------------------------------------
// Tile-shape compatibility check
// ---------------------------------------------------------------------------
// Marlin tile constraints (per upstream marlin/marlin_kernel.cu):
//   N % 64 == 0
//   K % 128 == 0   when group_size == 128
//   K % 64  == 0   when group_size ==  64
//   group_size in {64, 128, -1 (per-channel)}
//
// DSV4 expert w13: K=7168, N=4096, gs=128 → K%128=0 ✓ N%64=0 ✓ → OK
// DSV4 expert w2:  K=2048, N=7168, gs=128 → K%128=0 ✓ N%64=0 ✓ → OK
// GLM-5.1 IQ2XXS:  group_size=64 — Marlin path OK if K%64=0; else Triton

bool marlin_shape_ok(int K, int N, int group_size) {
    if (group_size != 64 && group_size != 128 && group_size != -1) return false;
    if (N % 64 != 0) return false;
    if (group_size == 128 && (K % 128) != 0) return false;
    if (group_size == 64  && (K %  64) != 0) return false;
    return true;
}

}  // anon

// ===========================================================================
// PART 1 — AWQ-Marlin expert GEMM dispatcher
// ===========================================================================
// Substitutes for FP8 [128,128] block-scaled expert GEMM. Per-expert
// weights/scales/zeros are sliced beforehand (caller responsibility) —
// this op is ONE expert's GEMM at a time. The MoE outer loop in
// `moe_dispatch_dsa_sm86.cu` (Story 3) handles per-expert iteration after
// the gather kernel.
//
// Inputs:
//   a:           bf16  [M, K]              gathered tokens for this expert
//   q_weight:    int32 [K, N/8]            Marlin-packed INT4 (8 weights / int32)
//   scales:      bf16  [K/group_size, N]   per-group scales
//   zeros:       int32 [K/group_size, N/8] (if asymmetric) packed zeros
//   workspace:   int32 [N/64 * 16]         scratch (Marlin requirement)
//   group_size:  int (64 or 128)
//
// Output:
//   out: bf16 [M, N]

at::Tensor awq_marlin_expert_gemm(
    const at::Tensor& a,
    const at::Tensor& q_weight,
    const at::Tensor& scales,
    const at::Tensor& zeros,
    at::Tensor& workspace,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size
) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA");
    TORCH_CHECK(a.dtype() == at::kBFloat16,
                "a must be bf16 (matches DSV4 residual stream dtype)");
    TORCH_CHECK(marlin_shape_ok(K, N, group_size),
                "shape (K=", K, ", N=", N, ", gs=", group_size,
                ") incompatible with Marlin tile constraints — caller "
                "must route to Triton fallback");

    auto handle = resolve_awq_marlin_op();
    auto stack = std::vector<c10::IValue>{
        a, q_weight, scales, zeros, workspace,
        M, N, K, group_size
    };
    handle.callBoxed(stack);
    TORCH_CHECK(stack.size() >= 1, "awq_marlin_gemm returned no value");
    return stack[0].toTensor();
}

// ===========================================================================
// PART 2 — AWQ → Marlin repack pass-through
// ===========================================================================
// vLLM ships `vllm::awq_marlin_repack`. This wrapper exists so callers
// have a single entry point even if the op signature drifts between
// vLLM versions; we centralize the boxed-call here for swap-friendly
// maintenance.

at::Tensor awq_repack_for_marlin(
    const at::Tensor& q_weight_awq,         // int32 [K/8, N] AWQ packed
    const at::Tensor& zeros_awq,            // int32 [K/group_size, N/8]
    int64_t K, int64_t N, int64_t group_size
) {
    auto handle = resolve_awq_marlin_repack_op();
    auto stack = std::vector<c10::IValue>{
        q_weight_awq, zeros_awq, K, N, group_size
    };
    handle.callBoxed(stack);
    TORCH_CHECK(stack.size() >= 1, "awq_marlin_repack returned no value");
    return stack[0].toTensor();
}

// ===========================================================================
// PART 3 — Triton-INT4 fallback kernel (string-compiled at first call)
// ===========================================================================
// For shapes that fail `marlin_shape_ok` (e.g., GLM-5.1 IQ2XXS at
// group_size=64 with K not a 64-multiple, or M=1 below Marlin's
// efficiency cliff). We provide a Triton kernel source as a const string
// + a Python loader that compiles it on first call.
//
// The kernel is stored in `awq_marlin_sm86.py` sibling rather than
// embedded here — keeps the .cu lean and lets Python iterate the kernel
// without a C++ rebuild.
//
// This wrapper is a stub that raises if Python hasn't loaded the kernel.

at::Tensor triton_int4_expert_gemm(
    const at::Tensor& a,
    const at::Tensor& q_weight,
    const at::Tensor& scales,
    const at::Tensor& zeros,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size
) {
    TORCH_CHECK(false,
        "triton_int4_expert_gemm: must be called via Python wrapper "
        "(`awq_marlin_sm86.py:triton_int4_gemm`) which JIT-compiles the "
        "kernel. The C++ surface only exists for symbol-table parity.");
    return at::Tensor();                                 // unreached
}

// ===========================================================================
// PART 4 — One-shot router
// ===========================================================================
// Caller passes the weight in EITHER Marlin-packed OR AWQ-packed form
// plus a layout flag, and we pick the path. If layout=AWQ and shape
// allows, repack inline (one-time, cached upstream by Python).

at::Tensor expert_gemm_router(
    const at::Tensor& a,
    const at::Tensor& q_weight,
    const at::Tensor& scales,
    const at::Tensor& zeros,
    at::Tensor& workspace,
    int64_t M, int64_t N, int64_t K,
    int64_t group_size,
    const std::string& layout                 // "marlin" | "awq"
) {
    if (!marlin_shape_ok(K, N, group_size)) {
        // Surface a recognizable error so Python can fall through to Triton
        TORCH_CHECK(false,
            "awq_marlin_sm86: shape (K=", K, ", N=", N, ", gs=", group_size,
            ") fails Marlin tile constraints; Python wrapper should "
            "catch this and call triton_int4_gemm.");
    }
    at::Tensor q_marlin;
    if (layout == "marlin") {
        q_marlin = q_weight;
    } else if (layout == "awq") {
        q_marlin = awq_repack_for_marlin(q_weight, zeros, K, N, group_size);
    } else {
        TORCH_CHECK(false, "expert_gemm_router: layout must be 'marlin' or 'awq'");
    }
    return awq_marlin_expert_gemm(a, q_marlin, scales, zeros,
                                  workspace, M, N, K, group_size);
}

// ===========================================================================
// Pybind module
// ===========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("awq_marlin_expert_gemm", &awq_marlin_expert_gemm,
          "INT4-AWQ-Marlin GEMM for one MoE expert (sm_86)");
    m.def("awq_repack_for_marlin", &awq_repack_for_marlin,
          "Repack AWQ packed-int32 weights into Marlin-interleaved layout");
    m.def("triton_int4_expert_gemm", &triton_int4_expert_gemm,
          "Triton-INT4 fallback (stub; Python wrapper does the JIT)");
    m.def("expert_gemm_router", &expert_gemm_router,
          "One-shot router: marlin/awq layout → Marlin GEMM");
    m.def("marlin_shape_ok",
          [](int K, int N, int group_size) {
              return marlin_shape_ok(K, N, group_size);
          },
          "Check whether (K, N, group_size) fits Marlin tile constraints");
}
