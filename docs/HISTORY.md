# History — How we got here

This is a research-systems engineering project conducted by ProtoAI (z / Bakari McCoy / IntuitIntel LLC) using a 9-agent Claude Code orchestration framework on a 16× RTX 3090 ZCCX cluster.

## Timeline

### 2026-04 early — vLLM 0.20.0 + PR #40760 lands DSA model classes
DeepSeek-V4-Flash and GLM-5.1-IQ2XXS become loadable. Both depend on the `DeepseekV4FlashMLASparseBackend` and `GLM5MoEModel` classes which route through `vllm/_C.abi3.so` Hopper-only kernels.

### 2026-04-23 — First DSV4 attempts on 16× RTX 3090
DSV4-Flash original (158 GiB FP4+FP8 mixed, [1,32] block scales): no Ampere kernel exists for [1,32] blocks. Switched to sgl-project/DeepSeek-V4-Flash-FP8 (274 GiB pure FP8, [128,128] blocks) → MARLIN auto-selects.

### 2026-04-25 → 2026-04-26 — The 17-patch cascade
After weights load, every cold POST `/v1/completions` hit a different assertion. Each fix unblocked one stage and revealed the next:

| # | Patch | What it fixed |
|---|---|---|
| 1 | patch_load_w13_diag.sh | Diagnostic prints in `_load_w13` shape mismatch |
| 2 | patch_dsv4_moe_block_v2 → reverted | Forced block_n=1, block_k=32 — broke sgl-fp8 |
| 3 | patch_dsv4_moe_block_v2_revert | Restored vanilla Fp8MoEMethod |
| 4 | patch_dsv4_attn_groups | `n_local_groups = max(1, ...)` for TP > o_groups=8 |
| 5 | patch_dsv4_supports_pp | Declared SupportsPP + stub `make_empty_intermediate_tensors` |
| 6 | patch_dsv4_pp_skip_nonlocal | Skip attn_sink/fallback for non-local layers |
| 7 | patch_dsv4_pp_skip_v2 / v3 | Skip stacked + expert paths before params_dict access |
| 8 | patch_pp_kv_skip_indexer | gpu_model_runner skip indexer pseudo-layers |
| 9 | patch_dsv4_swa_skip (v1) | Replace 1st `assert swa_metadata` with return |
| 10 | patch_dsv4_swa_skip_v2 | Replace 2nd `assert swa_metadata` (forward_context) |
| 11 | patch_skip_finfer_warmup | Skip FlashInfer mixed-batch warmup `_dummy_run` |
| 12 | patch_dsv4_dummy_short_circuit | Hoist dummy-run exit ABOVE indexer/compressor |
| 13 | patch_dsv4_indexer_bypass → reverted | Forced compressor-only path |
| 14 | patch_dsv4_compressor_bypass → reverted | Forced SWA-only path |
| 15 | patch_compressor_state_skip | `state_metadata.get()` + None return (graceful) |
| 16 | patch_sparse_indexer_stub | Replace `torch.ops.vllm.sparse_attn_indexer` with Python stub |
| 17 | YAML/launcher edits | profile rewrite, kernel-config, util/ctx/batch tuning |

**End state of cascade:** server boots, `/v1/models` 200 OK, `/health` 200 OK. `/v1/completions` returns 500 — bare assertion at `gpu_model_runner.py:4062 execute_model` inside compiled DSA kernels.

### 2026-04-26 — Decision: rebuild .so with sm_86 paths
This repo's mission begins. CC0 leads, CC1-CC9 workers each own a lane.

### 2026-04-27 → onwards — The rebuild

See `docs/02_KERNEL_INVENTORY.md` (CC1's output) for per-kernel scope.

## Forensics archive

The full pre-rebuild forensics live at:
- `~/AGENT/BENCH_MATRIX/DSV4_FULL_SUMMARY_2026-04-26.md` — every patch, every kernel try
- `~/AGENT/BENCH_MATRIX/DSV4_3090_BYPASS_PLAN.md` — verified PR #40760 hard-gate analysis
- `~/AGENT/research/DSV4_SM86_NOVEL_APPROACHES_20260425.md`
- `~/AGENT/research/DSV4_SM86_SHIM_FULL_REFERENCE_20260425.md`
- `~/AGENT/comms/DSV4_TRITON_FP8_EMULATION_RESEARCH.md`
- `~/AGENT/comms/DSV4_BF16_KV_ESCAPE_FINDINGS.md`
- `~/AGENT/comms/DSV4_8BIT_RED_TEAM_AGENT[1-3].md`

These will be consolidated into `patches/historical_cascade/` in this repo as a reference dump (NOT applied — historical only).

## The agentic framework

This project uses a custom 10-agent Claude Code orchestration:
- **CC0** = LEAD (orchestrator + cluster gatekeeper for L4 cycles)
- **CC1** = inventory (DSA kernel audit)
- **CC2** = build infrastructure (cmake + ABI alignment)
- **CC3** = sparse_attn_indexer port
- **CC4** = lightning bundle (compressor + mla_sparse + swa)
- **CC5** = MoE-DSA dispatch + TP=16 `o_groups` (HIGH-RISK isolated)
- **CC6** = correctness gate (L1-L5 harness)
- **CC7** = packaging + cluster fanout
- **CC8** = deploy + perf bench
- **CC9** = cluster broker (drain/nuke/L4 driver/artifact custodian)

All 10 agents run on z's ws10 mac with 130+ enforcement hooks, fleet MCP visibility (live cluster state), vecdb retrieval (BAAI/bge-small-en-v1.5 over 25K+ chunks), approval-gated cluster mutations, and per-agent tmux logging.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
