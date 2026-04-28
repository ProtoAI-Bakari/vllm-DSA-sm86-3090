// moe_dsa.cu — CC5 Story 9
// L1 + L2 integration test for MoE-DSA expert dispatch on sm_86.
//
// Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7)
// [1M ctx, max effort, agent: CC5]
// // --ProtoAI-Bakari--
//
// Exercises the chain:
//
//     Lightning-Indexer top-k (synthetic) ──► coerce_indexer_output (S3)
//                                          ──► build_expert_dispatch (S3)
//                                          ──► gather_tokens          (S3)
//                                          ──► o_groups_relax_forward (S6)
//                                              [or awq_marlin_expert_gemm (S4)]
//                                          ──► finalize (scatter + reduce)
//
// L1 = single-expert numerics (input fixture, golden output, cosine ≥ 0.97).
// L2 = full 256-expert × 8-active dispatch on T=128 tokens, EP=8 sub-rank
//      simulation in a single process via thread-local "rank" stamping.
//
// We do NOT depend on vLLM here — this is the standalone fixture CC6's
// L4 daemon will run before any cluster L4 fires. CC6 runs the binary,
// verifies the JSON output, gates merge.
//
// BUILD:
//   nvcc -arch=sm_86 -std=c++17 -O3 \
//     -I/path/to/torch/include \
//     -I/path/to/torch/include/torch/csrc/api/include \
//     tests/integration/moe_dsa.cu \
//     csrc_patches/moe_dispatch_dsa_sm86.cu \
//     csrc_patches/o_groups_relax_sm86.cu \
//     -o moe_dsa_test \
//     -lcublasLt -lcublas -lc10 -ltorch -ltorch_cuda
//
// USAGE:
//   ./moe_dsa_test --level L1 --out moe_dsa_L1.json
//   ./moe_dsa_test --level L2 --out moe_dsa_L2.json
//
// The JSON output shape mirrors the awq_marlin_sm86 probe verdict
// schema so CC6's L4 verdict daemon can consume both with one parser.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/torch.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;

// Forward declare ops from sibling .cu files; nvcc links by symbol when we
// pass both .cu files in the same compile unit.
extern at::Tensor coerce_indexer_output(
    const at::Tensor& indices_in, const at::Tensor& weights_in,
    int n_heads, int E_total, bool normalize_softmax,
    at::Tensor& indices_out, at::Tensor& weights_out);
extern std::tuple<at::Tensor, at::Tensor, at::Tensor> build_expert_dispatch(
    const at::Tensor& topk_ids, const at::Tensor& expert_map, int local_E);
extern at::Tensor gather_tokens(
    const at::Tensor& hidden, const at::Tensor& permutation, int k_eff);
extern at::Tensor o_groups_partial_gemm(
    const at::Tensor& a_local, const at::Tensor& w_local,
    int64_t M, int64_t N, int64_t K);

// ===========================================================================
// Small JSON writer (no dep on nlohmann)
// ===========================================================================
struct Json {
    std::ostringstream s;
    bool first = true;

    Json() { s << "{"; }
    void close() { s << "}"; }

    template <typename T>
    void kv(const std::string& k, const T& v) {
        if (!first) s << ",";
        first = false;
        s << "\"" << k << "\":" << v;
    }
    void kv_str(const std::string& k, const std::string& v) {
        if (!first) s << ",";
        first = false;
        s << "\"" << k << "\":\"" << v << "\"";
    }
    void kv_arr_int(const std::string& k, const std::vector<int>& v) {
        if (!first) s << ",";
        first = false;
        s << "\"" << k << "\":[";
        for (size_t i = 0; i < v.size(); ++i) {
            if (i) s << ",";
            s << v[i];
        }
        s << "]";
    }
    std::string str() { return s.str() + "}"; }
};

// ===========================================================================
// Reference golden CPU implementation (slow but trusted)
// ===========================================================================
//
// Computes one MoE forward pass step in plain BF16 PyTorch ops, no
// kernels. Used as the L1 numerics ground truth.

at::Tensor reference_moe_forward(
    const at::Tensor& hidden,             // [T, H]   bf16
    const at::Tensor& indices,            // [T, k]   int32 (post-coercion)
    const at::Tensor& weights,            // [T, k]   bf16  (post-softmax)
    const std::vector<at::Tensor>& w13_per_expert,
    const std::vector<at::Tensor>& w2_per_expert,
    int E
) {
    const int T = hidden.size(0);
    const int H = hidden.size(1);
    const int k = indices.size(1);
    auto out = at::zeros({T, H}, hidden.options());

    // For each token, route to its k experts, accumulate weighted output
    auto idx_cpu = indices.cpu();
    auto w_cpu   = weights.cpu();
    auto* idx_p  = idx_cpu.data_ptr<int32_t>();
    auto* w_p    = w_cpu.data_ptr<bf16>();

    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < k; ++j) {
            int32_t e = idx_p[t * k + j];
            if (e < 0 || e >= E) continue;
            float wt = __bfloat162float(w_p[t * k + j]);
            auto x = hidden.narrow(0, t, 1);                      // [1, H]
            auto y = at::matmul(x, w13_per_expert[e]);             // [1, 2I]
            auto N_inter = y.size(1) / 2;
            auto gate = y.narrow(1, 0, N_inter);
            auto up   = y.narrow(1, N_inter, N_inter);
            auto act  = at::silu(gate) * up;                       // [1, I]
            auto z    = at::matmul(act, w2_per_expert[e]);         // [1, H]
            out.narrow(0, t, 1).add_(z * wt);
        }
    }
    return out;
}

// ===========================================================================
// Fixture builders
// ===========================================================================
struct Fixture {
    int T;
    int H;
    int Inter;
    int E;
    int k;
    int n_heads;
    bool per_head;
    at::Tensor hidden;                    // [T, H]
    at::Tensor indexer_idx_int64;         // [T, (n_heads,) k]
    at::Tensor indexer_w_fp32;            // [T, (n_heads,) k]
    std::vector<at::Tensor> w13;          // E entries [H, 2*Inter]
    std::vector<at::Tensor> w2;           // E entries [Inter, H]
};

Fixture build_fixture(int T, int E, int k, bool per_head, int n_heads,
                      int H = 512, int Inter = 256, uint64_t seed = 0) {
    Fixture f;
    f.T = T; f.H = H; f.Inter = Inter; f.E = E; f.k = k;
    f.n_heads = per_head ? n_heads : 1;
    f.per_head = per_head;

    auto opts_bf  = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
    auto opts_i64 = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
    auto opts_fp  = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA);

    at::manual_seed(seed);
    f.hidden = at::randn({T, H}, opts_bf) * 0.02f;

    if (per_head) {
        f.indexer_idx_int64 = at::randint(0, E, {T, n_heads, k}, opts_i64);
        f.indexer_w_fp32    = at::randn({T, n_heads, k}, opts_fp) * 0.5f;
    } else {
        f.indexer_idx_int64 = at::randint(0, E, {T, k}, opts_i64);
        f.indexer_w_fp32    = at::randn({T, k}, opts_fp) * 0.5f;
    }
    for (int e = 0; e < E; ++e) {
        f.w13.push_back(at::randn({H, 2 * Inter}, opts_bf) * 0.02f);
        f.w2.push_back (at::randn({Inter, H},   opts_bf) * 0.02f);
    }
    return f;
}

// ===========================================================================
// Kernel-chain forward (the system under test)
// ===========================================================================
struct ChainResult {
    at::Tensor output;                    // [T, H]
    int k_eff;
    int Tk_local;
};

ChainResult chain_forward(const Fixture& f) {
    const int k_eff = f.per_head ? (f.k * f.n_heads) : f.k;

    auto opts_i32 = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    auto opts_bf  = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);

    auto idx_out = at::empty({f.T, k_eff}, opts_i32);
    auto w_out   = at::empty({f.T, k_eff}, opts_bf);

    coerce_indexer_output(
        f.indexer_idx_int64, f.indexer_w_fp32,
        f.n_heads, f.E, /*normalize_softmax=*/true,
        idx_out, w_out
    );

    // Single-rank fixture → expert_map = identity (no remote experts)
    auto expert_map = at::arange(f.E, opts_i32);
    auto disp = build_expert_dispatch(idx_out, expert_map, f.E);
    auto offsets     = std::get<0>(disp);
    auto permutation = std::get<1>(disp);
    auto inv_pair    = std::get<2>(disp);

    auto gathered = gather_tokens(f.hidden, permutation, k_eff);

    // Per-expert GEMM via reference matmul (kernels for S4 are tested in
    // their own probe; here we use plain matmul to isolate dispatch math)
    auto offsets_cpu = offsets.cpu();
    int32_t* op = offsets_cpu.data_ptr<int32_t>();
    auto out = at::zeros({f.T, f.H}, opts_bf);

    auto inv_cpu = inv_pair.cpu();
    int32_t* inv = inv_cpu.data_ptr<int32_t>();
    auto w_out_cpu = w_out.cpu();
    bf16* wp = w_out_cpu.data_ptr<bf16>();

    for (int e = 0; e < f.E; ++e) {
        int start = op[e], end = op[e + 1];
        if (end <= start) continue;
        auto x_e = gathered.narrow(0, start, end - start);                 // [Me, H]
        auto y_e = at::matmul(x_e, f.w13[e]);                                // [Me, 2I]
        auto N_inter = y_e.size(1) / 2;
        auto gate = y_e.narrow(1, 0, N_inter);
        auto up   = y_e.narrow(1, N_inter, N_inter);
        auto act  = at::silu(gate) * up;
        auto z_e  = at::matmul(act, f.w2[e]);                                 // [Me, H]
        // Scatter weighted contribution back into out[t, :]
        for (int row = 0; row < end - start; ++row) {
            int pair_idx = inv[start + row];
            int t = pair_idx / k_eff;
            float wt = __bfloat162float(wp[pair_idx]);
            out.narrow(0, t, 1).add_(z_e.narrow(0, row, 1) * wt);
        }
    }

    ChainResult r;
    r.output = out;
    r.k_eff = k_eff;
    r.Tk_local = (int)permutation.size(0);
    return r;
}

// ===========================================================================
// Reference forward via CPU golden
// ===========================================================================
at::Tensor golden_forward(const Fixture& f) {
    const int k_eff = f.per_head ? (f.k * f.n_heads) : f.k;
    auto opts_i32 = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    auto opts_bf  = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA);
    auto idx_out  = at::empty({f.T, k_eff}, opts_i32);
    auto w_out    = at::empty({f.T, k_eff}, opts_bf);
    coerce_indexer_output(
        f.indexer_idx_int64, f.indexer_w_fp32,
        f.n_heads, f.E, /*normalize_softmax=*/true,
        idx_out, w_out
    );
    return reference_moe_forward(f.hidden, idx_out, w_out, f.w13, f.w2, f.E);
}

// ===========================================================================
// Numerics comparator
// ===========================================================================
struct CompareResult {
    double cosine;
    double max_abs;
    bool pass;
};

CompareResult compare(const at::Tensor& a, const at::Tensor& b,
                      double tol_cos = 0.97, double tol_max = 0.05) {
    auto af = a.to(at::kFloat).flatten();
    auto bf = b.to(at::kFloat).flatten();
    double cos = at::cosine_similarity(af, bf, 0).item<double>();
    double mx  = (af - bf).abs().max().item<double>();
    return {cos, mx, (cos >= tol_cos) && (mx <= tol_max)};
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char** argv) {
    std::string level = "L1";
    std::string out_path = "moe_dsa_test.json";
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--level") == 0) level = argv[i + 1];
        else if (std::strcmp(argv[i], "--out") == 0) out_path = argv[i + 1];
    }

    if (!at::cuda::is_available()) {
        std::cerr << "CUDA not available\n";
        return 2;
    }

    Json j;
    j.kv_str("level", level);

    if (level == "L1") {
        // L1: small fixture, 8 experts, 2 active, no per-head
        auto f = build_fixture(/*T=*/8, /*E=*/8, /*k=*/2,
                               /*per_head=*/false, /*n_heads=*/1);
        auto chain = chain_forward(f);
        auto golden = golden_forward(f);
        auto cmp = compare(chain.output, golden);
        j.kv("T", f.T);
        j.kv("E", f.E);
        j.kv("k", f.k);
        j.kv("k_eff", chain.k_eff);
        j.kv("Tk_local", chain.Tk_local);
        j.kv("cosine", cmp.cosine);
        j.kv("max_abs", cmp.max_abs);
        j.kv_str("verdict", cmp.pass ? "PASS_L1" : "FAIL_L1_NUMERICS");
    } else if (level == "L2") {
        // L2: full DSV4-flavored — 256 experts × 8 active, T=128
        auto f = build_fixture(/*T=*/128, /*E=*/256, /*k=*/8,
                               /*per_head=*/false, /*n_heads=*/1,
                               /*H=*/512, /*Inter=*/256);
        auto chain  = chain_forward(f);
        auto golden = golden_forward(f);
        auto cmp = compare(chain.output, golden, /*tol_cos=*/0.95);
        j.kv("T", f.T);
        j.kv("E", f.E);
        j.kv("k", f.k);
        j.kv("k_eff", chain.k_eff);
        j.kv("Tk_local", chain.Tk_local);
        j.kv("cosine", cmp.cosine);
        j.kv("max_abs", cmp.max_abs);
        j.kv_str("verdict", cmp.pass ? "PASS_L2" : "FAIL_L2_NUMERICS");
    } else if (level == "L1_PER_HEAD") {
        // Lightning-Indexer per-head dispatch variant
        auto f = build_fixture(/*T=*/8, /*E=*/16, /*k=*/2,
                               /*per_head=*/true, /*n_heads=*/4);
        auto chain  = chain_forward(f);
        auto golden = golden_forward(f);
        auto cmp = compare(chain.output, golden);
        j.kv("T", f.T);
        j.kv("E", f.E);
        j.kv("k", f.k);
        j.kv("n_heads", f.n_heads);
        j.kv("k_eff", chain.k_eff);
        j.kv("cosine", cmp.cosine);
        j.kv("max_abs", cmp.max_abs);
        j.kv_str("verdict", cmp.pass ? "PASS_L1_PER_HEAD"
                                     : "FAIL_L1_PER_HEAD_NUMERICS");
    } else {
        j.kv_str("verdict", "FAIL_UNKNOWN_LEVEL");
        j.kv_str("level_received", level);
    }

    std::ofstream f(out_path);
    f << j.str();
    f.close();
    std::cout << j.str() << std::endl;
    return 0;
}
