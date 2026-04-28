// moe_dispatch_dsa_sm86.cu — CC5 Story 3
// MoE-DSA routing patch for sm_86: handle 256 experts × 8 active per token.
//
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7)
// [1M ctx, max effort, agent: CC5]
// // --ProtoAI-Bakari--
//
// Why this file exists
// --------------------
// DSA replaces vLLM's standard MoE router with the Lightning Indexer's top-k
// pick. Downstream `MoEPrepareAndFinalizeNoDPEPModular` expects a strict
// `(topk_ids: int32[T,k], topk_weights: bf16[T,k])` contract. Lightning
// Indexer outputs do not match cleanly:
//
//   - indices may ship as int64
//   - weights may not be softmax-normalized
//   - per-head dispatch (shape `[T, n_heads, k]`) on some DSV4 variants
//
// On Hopper this is hidden by TMA + WGMMA fused kernels. On Ampere (sm_86)
// vLLM dispatches through the modular path that asserts on shape mismatch.
//
// This patch provides three small kernels + C++ wrappers:
//
//   1. coerce_indexer_output_kernel
//        int64 → int32 cast, optional per-token softmax normalize, optional
//        per-head flatten + weight rescale. Single fused pass.
//
//   2. build_expert_offsets_kernel
//        Given coerced topk_ids and an expert_map (global id → local slot or
//        -1), produce per-local-expert offset table (CSR start[]) plus the
//        gather permutation that lays tokens out in expert-major order.
//
//   3. gather_tokens_kernel
//        Permute hidden_states [T, H] into [sum_local M_e, H] using the
//        permutation from kernel 2. bf16 stores; H is multiple of 8 so we
//        do 16-byte vectorized loads via float4.
//
// Targets sm_80+ primitives only:
//   - cp.async.cg replaces TMA bulk loads (we use plain global loads here
//     since the gather is bandwidth-bound; cp.async would only help with
//     overlap, addressed in Story 7 EPLB tune)
//   - mma.sync warp-level (not needed in this patch — pure data movement)
//   - cooperative_groups for block-level reductions
//
// Build (CC2 cmake -DTORCH_CUDA_ARCH_LIST="8.6"):
//   /usr/local/cuda/bin/nvcc -arch=sm_86 -std=c++17 -O3 \
//     -Xcompiler=-fPIC -shared moe_dispatch_dsa_sm86.cu \
//     -o moe_dispatch_dsa_sm86.so $(python3-config --includes) \
//     -I$(python3 -c 'import torch.utils.cpp_extension as e; print(e.include_paths()[0])')
//
// Test surface: Story 9 (tests/integration/moe_dsa.cu) exercises end-to-end.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <torch/extension.h>

namespace cg = cooperative_groups;
using bf16 = __nv_bfloat16;

// ---------------------------------------------------------------------------
// Kernel 1 — coerce Lightning Indexer output to vLLM PREPARE contract
// ---------------------------------------------------------------------------
//
// Input variants supported (run-time switches, not compile-time):
//   indices_in:   int64 [T, k]   OR   int64 [T, n_heads, k]   (per-head)
//   weights_in:   fp32  [T, k]   OR   fp32  [T, n_heads, k]   (raw scores)
//
// Output:
//   indices_out:  int32 [T, k_eff]
//   weights_out:  bf16  [T, k_eff]
//
// k_eff = k                     when per_head == false
// k_eff = k * n_heads           when per_head == true (we flatten + rescale)
//
// Behavior:
//   - cast int64 → int32 with overflow guard (E < 2^31, holds for E=256)
//   - if normalize_softmax: per-token softmax over the k_eff dim
//   - if per_head: flatten the heads dim and divide weights by n_heads to
//     preserve total contribution mass
//
// Shape invariant: T * k_eff <= 2^24 for our launches (block per token,
// k_eff threads per block)

template <bool PerHead, bool NormalizeSoftmax>
__global__ void coerce_indexer_output_kernel(
    const int64_t* __restrict__ indices_in,    // [T, n_heads, k] or [T, k]
    const float*   __restrict__ weights_in,    // same layout
    int32_t*       __restrict__ indices_out,   // [T, k_eff]
    bf16*          __restrict__ weights_out,   // [T, k_eff]
    int T,
    int k,
    int n_heads,                               // 1 if !PerHead
    int E_total                                // for overflow check; E=256 typ
) {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int k_eff = PerHead ? (k * n_heads) : k;
    const int tid = threadIdx.x;

    // Each thread handles a strided slice of k_eff
    extern __shared__ float smem[];          // size = k_eff floats for softmax

    // Stage 1: cast + (optional) per-head rescale, write into smem (raw scores)
    for (int j = tid; j < k_eff; j += blockDim.x) {
        int src_idx;
        if (PerHead) {
            // src laid out as [T, n_heads, k]; flatten head*k + inner
            src_idx = t * (n_heads * k) + j;
        } else {
            src_idx = t * k + j;
        }
        int64_t exp_id_64 = indices_in[src_idx];
        float w           = weights_in[src_idx];
        if (PerHead) {
            // preserve total weight mass after head-flatten
            w = w / static_cast<float>(n_heads);
        }
        // overflow guard — DSV4 + GLM-5.1 both have E=256, plenty of headroom
        int32_t exp_id_32 = (exp_id_64 >= 0 && exp_id_64 < E_total)
                              ? static_cast<int32_t>(exp_id_64)
                              : -1;
        indices_out[t * k_eff + j] = exp_id_32;
        smem[j] = w;
    }
    __syncthreads();

    if (NormalizeSoftmax) {
        // Block-wide softmax over k_eff (typ k=8, k_eff<=64 for per-head)
        // Stage A: max
        float local_max = -INFINITY;
        for (int j = tid; j < k_eff; j += blockDim.x) {
            local_max = fmaxf(local_max, smem[j]);
        }
        cg::thread_block block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        float warp_max = cg::reduce(warp, local_max, cg::greater<float>());
        // For block sizes <= 32 (we launch with min(k_eff, 32)) warp_max is
        // already the block max; otherwise fall back to a one-thread shuffle
        __shared__ float blk_max;
        if (warp.thread_rank() == 0) blk_max = warp_max;
        __syncthreads();

        // Stage B: exp + sum
        float local_sum = 0.0f;
        for (int j = tid; j < k_eff; j += blockDim.x) {
            float e = expf(smem[j] - blk_max);
            smem[j] = e;
            local_sum += e;
        }
        float warp_sum = cg::reduce(warp, local_sum, cg::plus<float>());
        __shared__ float blk_sum;
        if (warp.thread_rank() == 0) blk_sum = warp_sum;
        __syncthreads();
        const float inv_sum = 1.0f / fmaxf(blk_sum, 1e-20f);

        // Stage C: write normalized bf16
        for (int j = tid; j < k_eff; j += blockDim.x) {
            float w = smem[j] * inv_sum;
            weights_out[t * k_eff + j] = __float2bfloat16(w);
        }
    } else {
        // No softmax: just bf16-cast the smem scores
        for (int j = tid; j < k_eff; j += blockDim.x) {
            weights_out[t * k_eff + j] = __float2bfloat16(smem[j]);
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2 — build expert offsets + gather permutation
// ---------------------------------------------------------------------------
//
// Inputs:
//   topk_ids_int32: [T, k]      coerced from kernel 1
//   expert_map:     [E_total]   global id → local slot idx (0..local_E-1)
//                                or -1 if remote
//
// Outputs:
//   counts_per_expert: [local_E]            atomic-incremented count
//   permutation:       [T*k]                token-index in gather order
//   slot_per_pair:     [T*k]                per-pair local slot (or -1 remote)
//
// CSR offsets are then computed on host via exclusive scan of counts.
// (We could fuse via cub::DeviceScan but for local_E=32 the host scan is
// trivially cheap and avoids a sync trip.)
//
// Two-pass design:
//   Pass A: count tokens per local expert (atomic add)
//   Pass B: assign each (t,k) pair to its slot in the permutation
//
// Pass B uses a per-expert running-cursor (also atomic add) to assign a
// unique gather index. The gather order is stable per-expert but not
// strictly token-order; PREPARE downstream does not require token-order.

__global__ void count_expert_tokens_kernel(
    const int32_t* __restrict__ topk_ids,      // [T*k]
    const int32_t* __restrict__ expert_map,    // [E_total]
    int32_t*       __restrict__ counts,        // [local_E]
    int32_t*       __restrict__ slot_per_pair, // [T*k]
    int Tk,
    int E_total,
    int local_E
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Tk) return;
    int32_t exp_id = topk_ids[idx];
    int32_t slot = (exp_id >= 0 && exp_id < E_total) ? expert_map[exp_id] : -1;
    slot_per_pair[idx] = slot;
    if (slot >= 0 && slot < local_E) {
        atomicAdd(&counts[slot], 1);
    }
}

__global__ void build_permutation_kernel(
    const int32_t* __restrict__ slot_per_pair, // [T*k]
    const int32_t* __restrict__ offsets,       // [local_E + 1]
    int32_t*       __restrict__ cursors,       // [local_E] (zero-init)
    int32_t*       __restrict__ permutation,   // [Tk_local]
    int32_t*       __restrict__ inv_pair_ids,  // [Tk_local] back-pointer
    int Tk,
    int local_E
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Tk) return;
    int32_t slot = slot_per_pair[idx];
    if (slot < 0 || slot >= local_E) return;
    int32_t pos = offsets[slot] + atomicAdd(&cursors[slot], 1);
    permutation[pos] = idx / 1;             // pair index (t*k + j); the
                                            //   token index is idx/k_eff and
                                            //   the k-slot is idx%k_eff
    inv_pair_ids[pos] = idx;
}

// ---------------------------------------------------------------------------
// Kernel 3 — gather tokens into expert-major layout
// ---------------------------------------------------------------------------
//
// Input:
//   hidden_states: bf16 [T, H]
//   permutation:   int32 [Tk_local]   pair_idx = t*k_eff + j; gather t = pair_idx/k_eff
//   k_eff
//
// Output:
//   gathered: bf16 [Tk_local, H]
//
// One block per gathered row. Each block does H/8 16-byte loads via float4.
// H must be a multiple of 8 (DSV4 H=7168 → 7168/8=896 ✓).

__global__ void gather_tokens_kernel(
    const bf16*    __restrict__ hidden,       // [T, H]
    const int32_t* __restrict__ permutation,  // [Tk_local]
    bf16*          __restrict__ gathered,     // [Tk_local, H]
    int H,
    int k_eff
) {
    const int row_out = blockIdx.x;
    const int pair_idx = permutation[row_out];
    const int t = pair_idx / k_eff;
    const float4* src4 =
        reinterpret_cast<const float4*>(hidden + t * H);
    float4* dst4 =
        reinterpret_cast<float4*>(gathered + row_out * H);
    const int H8 = H / 8;                     // 8 bf16 = 16 B = 1 float4
    for (int j = threadIdx.x; j < H8; j += blockDim.x) {
        dst4[j] = src4[j];
    }
}

// ---------------------------------------------------------------------------
// C++ wrappers (PyTorch extension)
// ---------------------------------------------------------------------------

at::Tensor coerce_indexer_output(
    const at::Tensor& indices_in,
    const at::Tensor& weights_in,
    int n_heads,
    int E_total,
    bool normalize_softmax,
    at::Tensor& indices_out,
    at::Tensor& weights_out
) {
    TORCH_CHECK(indices_in.is_cuda() && weights_in.is_cuda(),
                "coerce_indexer_output: tensors must be CUDA");
    TORCH_CHECK(indices_in.dtype() == at::kLong,
                "coerce_indexer_output: indices_in must be int64");
    TORCH_CHECK(weights_in.dtype() == at::kFloat,
                "coerce_indexer_output: weights_in must be float32");
    TORCH_CHECK(indices_out.dtype() == at::kInt,
                "coerce_indexer_output: indices_out must be int32");
    TORCH_CHECK(weights_out.dtype() == at::kBFloat16,
                "coerce_indexer_output: weights_out must be bfloat16");

    const bool per_head = (n_heads > 1);
    const int T = indices_in.size(0);
    const int k = per_head ? indices_in.size(2) : indices_in.size(1);
    const int k_eff = per_head ? (k * n_heads) : k;

    const int threads = (k_eff <= 32) ? k_eff : 32;
    const int blocks  = T;
    const size_t smem = k_eff * sizeof(float);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto kernel) {
        kernel<<<blocks, threads, smem, stream>>>(
            indices_in.data_ptr<int64_t>(),
            weights_in.data_ptr<float>(),
            indices_out.data_ptr<int32_t>(),
            weights_out.data_ptr<bf16>(),
            T, k, n_heads, E_total
        );
    };
    if (per_head && normalize_softmax)
        launch(coerce_indexer_output_kernel<true, true>);
    else if (per_head && !normalize_softmax)
        launch(coerce_indexer_output_kernel<true, false>);
    else if (!per_head && normalize_softmax)
        launch(coerce_indexer_output_kernel<false, true>);
    else
        launch(coerce_indexer_output_kernel<false, false>);
    AT_CUDA_CHECK(cudaGetLastError());
    return indices_out;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> build_expert_dispatch(
    const at::Tensor& topk_ids,        // int32 [T, k_eff] (coerced)
    const at::Tensor& expert_map,      // int32 [E_total]
    int local_E
) {
    TORCH_CHECK(topk_ids.is_cuda() && expert_map.is_cuda(),
                "build_expert_dispatch: tensors must be CUDA");
    TORCH_CHECK(topk_ids.dtype() == at::kInt,
                "build_expert_dispatch: topk_ids must be int32");
    TORCH_CHECK(expert_map.dtype() == at::kInt,
                "build_expert_dispatch: expert_map must be int32");

    const int T = topk_ids.size(0);
    const int k_eff = topk_ids.size(1);
    const int Tk = T * k_eff;
    const int E_total = expert_map.size(0);
    auto opts_i32 = topk_ids.options();

    auto counts        = at::zeros({local_E}, opts_i32);
    auto slot_per_pair = at::empty({Tk}, opts_i32);
    auto cursors       = at::zeros({local_E}, opts_i32);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        const int threads = 256;
        const int blocks  = (Tk + threads - 1) / threads;
        count_expert_tokens_kernel<<<blocks, threads, 0, stream>>>(
            topk_ids.data_ptr<int32_t>(),
            expert_map.data_ptr<int32_t>(),
            counts.data_ptr<int32_t>(),
            slot_per_pair.data_ptr<int32_t>(),
            Tk, E_total, local_E
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }

    // Exclusive scan on host (local_E ~32, trivial)
    auto counts_cpu = counts.cpu();
    auto offsets_cpu = at::zeros({local_E + 1}, counts_cpu.options());
    int32_t* c = counts_cpu.data_ptr<int32_t>();
    int32_t* o = offsets_cpu.data_ptr<int32_t>();
    int32_t running = 0;
    for (int i = 0; i < local_E; ++i) {
        o[i] = running;
        running += c[i];
    }
    o[local_E] = running;
    const int Tk_local = running;
    auto offsets = offsets_cpu.to(topk_ids.device());

    auto permutation  = at::empty({Tk_local}, opts_i32);
    auto inv_pair_ids = at::empty({Tk_local}, opts_i32);

    {
        const int threads = 256;
        const int blocks  = (Tk + threads - 1) / threads;
        build_permutation_kernel<<<blocks, threads, 0, stream>>>(
            slot_per_pair.data_ptr<int32_t>(),
            offsets.data_ptr<int32_t>(),
            cursors.data_ptr<int32_t>(),
            permutation.data_ptr<int32_t>(),
            inv_pair_ids.data_ptr<int32_t>(),
            Tk, local_E
        );
        AT_CUDA_CHECK(cudaGetLastError());
    }

    return std::make_tuple(offsets, permutation, inv_pair_ids);
}

at::Tensor gather_tokens(
    const at::Tensor& hidden,         // bf16 [T, H]
    const at::Tensor& permutation,    // int32 [Tk_local]
    int k_eff
) {
    TORCH_CHECK(hidden.is_cuda() && permutation.is_cuda(),
                "gather_tokens: tensors must be CUDA");
    TORCH_CHECK(hidden.dtype() == at::kBFloat16,
                "gather_tokens: hidden must be bfloat16");
    TORCH_CHECK(hidden.size(1) % 8 == 0,
                "gather_tokens: H must be multiple of 8 for vectorized gather");

    const int H = hidden.size(1);
    const int Tk_local = permutation.size(0);
    auto gathered = at::empty({Tk_local, H}, hidden.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int threads = 128;                  // 128 * 8 bf16 = 1 KiB per pass
    const int blocks  = Tk_local;
    gather_tokens_kernel<<<blocks, threads, 0, stream>>>(
        hidden.data_ptr<bf16>(),
        permutation.data_ptr<int32_t>(),
        gathered.data_ptr<bf16>(),
        H, k_eff
    );
    AT_CUDA_CHECK(cudaGetLastError());
    return gathered;
}

// ---------------------------------------------------------------------------
// Pybind11 module — wires into vllm._custom_ops style namespace
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coerce_indexer_output", &coerce_indexer_output,
          "DSA Lightning-Indexer output → vLLM PREPARE contract (int32 + bf16 softmax-normalized)");
    m.def("build_expert_dispatch", &build_expert_dispatch,
          "Build per-local-expert offsets + gather permutation for 256E/8A dispatch");
    m.def("gather_tokens", &gather_tokens,
          "Gather hidden_states into expert-major layout for MoE GEMM");
}
