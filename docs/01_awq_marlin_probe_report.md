# Story 1 — INT4-AWQ-Marlin sm_86 Viability Probe Report

**Track:** B (INT4 stairstep)
**Lane:** `lane-cc5-moe-dsa`
**Probe SHA:** `474dccd`
**Probe path:** `patches/01_dsa_sm86_kernels/awq_marlin_sm86_probe.py`
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC5]**

## 1. The hour-1 question

> Does Marlin (already shipped in vLLM for sm_86 W8A8) cover the INT4-AWQ
> dispatch path needed to substitute for DSV4-Flash-FP8's `[128,128]`
> block-scaled FP8 GEMM on Ampere — across a **256-expert MoE** with
> **8 active experts per token**?

If yes → Track B unlocks Bronze-Silver perf without writing new CUDA.
If partial → write a thin adapter (`moe_dispatch_dsa/awq_marlin_adapter_sm86.py`).
If no → fall back to Triton-INT4 lane (Story 4 backup).

## 2. Probe surface (what the script measures)

The probe exercises **op-resolution + shape acceptance**, not full numerics.
Full numerics need real AWQ-quantized expert weights, which require CC1's
calibration pipeline (Story 5). Hour-1 gate is structural.

| Layer | What probe checks | Verdict label |
|---|---|---|
| `torch.ops.vllm.awq_marlin_gemm` resolves | Op exists in installed vLLM build | `awq_marlin_resolved=true` |
| `torch.ops.vllm.awq_marlin_repack` resolves | Repack kernel callable | `repack_resolved=true` |
| Device cap = sm_86 | Confirms we are on the target arch | `sm86_confirmed=true` |
| DSV4 expert shapes accepted | M={1,4} K=7168 N=4096 (w13) + K=2048 N=7168 (w2) | `candidate_ok=true` per fixture |
| BF16 reference latency captured | Establishes "what we beat to be useful" | `bf16_ms` per fixture |

Verdicts:

| Code | Meaning | Next action |
|---|---|---|
| `PASS_OP_RESOLVED_AWAITING_CALIBRATION` | Marlin op available on sm_86, shapes accepted, full numerics deferred | **Proceed to Story 5** AWQ calibration → Story 4 backport |
| `FAIL_OP_UNRESOLVED` | `awq_marlin_gemm` not in installed vLLM | Block on **CC2** ABI alignment + reinstall before Story 4 |
| `FAIL_KERNEL_RUN` | Op resolved but threw on shape | Write adapter in `moe_dispatch_dsa/` (Story 4 sub-task) |
| `FAIL_NO_CUDA` / `FAIL_IMPORT` | Environment not ready | Block on CC2 venv + CC7 fanout |
| `FAIL_UNCAUGHT` | Unexpected | Investigate per-traceback |

## 3. Why the probe is structural, not semantic

DSV4 ships **pure FP8 [128,128] block-scaled** weights — there is no
INT4-AWQ checkpoint for it on HF today. Producing one requires:

1. Per-expert calibration on CC6's 1500-prompt set (256 experts × 8 active
   makes per-expert calibration the MoE-quant edge case — naive global
   calibration drops accuracy hard).
2. AWQ scale + zero-point packing into Marlin's expected layout.
3. Repack via `awq_marlin_repack` to the kernel's interleaved storage.

That pipeline is **Story 5 work**, not hour-1. The hour-1 win is to
**de-risk the kernel layer** so calibration work has a known-good target.

## 4. Decision matrix (what CC0 reads from probe verdict)

| Verdict | Track-B continues? | Track-A scope |
|---|---|---|
| `PASS_OP_RESOLVED_AWAITING_CALIBRATION` | **Yes — Story 5 unblocks** | Continue Story 2 audit in parallel |
| `FAIL_OP_UNRESOLVED` | Pause until CC2 reinstalls vLLM with AWQ-Marlin compiled | Track A (Stories 2-3) becomes critical path |
| `FAIL_KERNEL_RUN` | Adapter sub-task added before Story 4 | Track A unaffected |

## 5. Expected verdict on cuda5 (current cluster state)

Best estimate ahead of CC9 fire:

- vLLM `0.20.0+cu128` shipped with the AWQ-Marlin module under
  `vllm.model_executor.layers.quantization.awq_marlin` (well-established
  sm_80+ Ampere path). High prior on `awq_marlin_resolved=true`.
- The 17-patch cascade was on the **DSA attention path**, not the
  AWQ-Marlin GEMM path — those compiled units are independent.
- Most likely outcome: `PASS_OP_RESOLVED_AWAITING_CALIBRATION`, fixtures
  log `repack_resolved=true` with deferred numerics.

If that holds, Track B has a green light into Story 5.

## 6. Probe re-run protocol

After CC2 ships any vLLM ABI rebuild that touches MoE / quantization:

```bash
bash ~/AGENT/tools/cc_send.sh WC5 \
  "bash ~/AGENT/tools/ssh_node.sh cuda5 \
     'cd /repo/INSTALLERS/vllm-DSA-sm86-3090 && \
      time python3 patches/01_dsa_sm86_kernels/awq_marlin_sm86_probe.py'"
```

Verdict JSON lands at `cuda5:./awq_marlin_sm86_probe_<host>.json`. CC9
mirrors back to ws10 for CC0/CC6 review.

## 7. Open questions surfaced by probe design

1. **Group size = 128 vs 64.** DSV4's `[128,128]` block scale aligns to
   Marlin group_size=128 by construction. GLM-5.1 IQ2XXS uses group_size=64
   in its quantization metadata. Story 4 must handle both.
2. **Asymmetric vs symmetric AWQ.** vLLM's AWQ-Marlin assumes asymmetric
   (zero + scale per group). DSV4's FP8 block is symmetric. Calibration in
   Story 5 must produce asymmetric weights.
3. **Per-expert workspace allocation.** Marlin needs a scratch tensor;
   under EP=8, scratch per local-expert × max-tokens-per-expert. Sizing
   review goes in Story 4.

## 8. Closure

Story 1 ships:
- `patches/01_dsa_sm86_kernels/awq_marlin_sm86_probe.py` (probe, 345 LOC)
- `docs/01_awq_marlin_probe_report.md` (this file)

Track B ready to continue at Story 5 once L4 verdict confirms
`PASS_OP_RESOLVED_AWAITING_CALIBRATION`.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC5]
