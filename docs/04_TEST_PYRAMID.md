# 04 — Test Pyramid

CC6 (gate lane) owns this harness. The pyramid is L1 → L5 in increasing hardware/wall cost. Every accepted kernel merge passes through ALL applicable tiers.

## L1 — Unit (single 3090, seconds)

Per-kernel bit-correctness. Input tensors → output tensors → numerical bit-compare against a reference.

```bash
bash tests/unit/run_kernel.sh sparse_attn_indexer
```

Reference selection:
- if a Hopper machine produced the reference: bit-exact within fp16/bf16 tolerance
- otherwise: derive a deterministic reference from the algorithm (online softmax + top-k indexer) implemented in pure pytorch on CPU

## L2 — Forward slice (1 node, 1-2 min)

Kernel inside its surrounding stack at TP=2 on a single Mac Pro node (2× 3090 NVLink). Catches stride / dtype / memory-layout integration bugs that L1 misses.

```bash
bash tests/integration/run_l2_forward_slice.sh sparse_attn_indexer
```

## L3 — Small-model load (1-2 nodes, ~5 min)

Full vLLM request path on a tiny test fixture (small DSA-class model or a stripped-down DSV4 with reduced layer count). Validates the kernel works through actual `engine.generate()`.

```bash
bash tests/integration/run_l3_small_model.sh
```

## L4 — Full-cluster integration (8 nodes, drained+nuked, 10-15 min)

The actual model: DSV4-Flash-FP8 at TP=2 × EP=8 OR PP=2+TP=8 fallback. Real prompts. Real tokens. The ONLY level that surfaces deeper Hopper-only kernels we haven't ported yet.

```bash
bash tests/correctness/run_l4_full_cluster.sh dsv4
bash tests/correctness/run_l4_full_cluster.sh glm51
```

## L5 — Perf bench (8 nodes, 30-60 min)

Concurrency sweep + NCCL fabric profile + numerics retention against the llama.cpp PP baseline.

```bash
bash bench/runners/run_concurrency_sweep.sh dsv4
```

## Numerics gate

Every L4 run produces:
- top-1 token agreement vs llama.cpp PP baseline (target ≥98%)
- logit cosine (target ≥0.97)
- perplexity drop on MMLU subset (target ≤2%)

CC6 captures these to `bench/results/<date>/numerics_<model>.json`. CC0 reverts any merge that fails any gate.

## llama.cpp PP baseline strategy

z's GLM-5.1 PP runs at ~7 t/s conc=1, ~25 t/s at conc=4-6. Sequential 1500-prompt baseline = days. **Capture in micro-batches at conc=4-6** to leverage the cluster's parallel headroom: 1500 prompts ÷ 6 parallel ≈ 250 sequential equivalents at 25 t/s ≈ a few hours.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
