# CC5 Story Backlog — MoE-DSA dispatch + INT4-AWQ-Marlin (dual track)
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC5]**
**Pruned 2026-04-28 per CC0 directive — every story traces to PROJECT GOAL: DSV4 + GLM-5.1 coherent tokens at perf tier**

Dual-track per LEAD's INT4 staircase. Track A = MoE-DSA dispatch (unblocks EP=8). Track B = INT4-AWQ-Marlin sm_86 (unblocks Bronze-Silver perf).

| # | Story | Track | Complexity | Output | Status |
|---|---|---|---|---|---|
| 1 | Hour-1 PROBE: INT4-AWQ-Marlin sm_86 viability — does Marlin already serve a 256-expert MoE on Ampere? | B | patches/01_dsa_sm86_kernels/awq_marlin_sm86_probe.py + report | DONE 474dccd + 8a895ab |
| 2 | MoEPrepareAndFinalizeNoDPEPModular audit — verify EP=8 dispatch shape works on rebuilt vLLM | A | docs/02_moe_dsa_dispatch_audit.md | DONE 8a895ab |
| 3 | MoE-DSA routing patch — handle 256 experts × 8 active per token | A | csrc_patches/moe_dispatch_dsa_sm86.cu | next |
| 4 | INT4-AWQ-Marlin sm_86 backport — bridge Marlin W8A8 (existing sm_86) to DSV4's INT4-AWQ; Triton-INT4 backup | B | csrc_patches/awq_marlin_sm86.cu | next |
| 5 | Per-expert MoE-aware AWQ calibration on CC6's prompt set | B | scripts/awq_calibrate_dsv4.py | next |
| 6 | TP=16 o_groups=8 relaxation (HIGH-RISK isolated lane — gated on PP=2+TP=8 shipping clean first) | A | csrc_patches/o_groups_relax.cu | gated |
| 7 | EPLB (Expert Parallel Load Balancing) — stretch goal, post first coherent token | A | csrc_patches/eplb_sm86.cu | stretch (confirmed) |
| 8 | MTP (multi-token prediction) head wiring for spec-decode — stretch, post first coherent token | B | csrc_patches/mtp_head.cu | stretch (confirmed) |
| 9 | L1 + L2 integration test for MoE-DSA expert dispatch | A | tests/integration/moe_dsa.cu | next |

## Per-story rationale → project goal trace

- **#1 AWQ-Marlin viability probe** — rationale: gates Track B; without proven Marlin op-resolve on sm_86 the INT4 path cannot substitute for DSV4's Hopper-only FP8 GEMM, leaving DSV4 at 10-20 t/s BF16 ceiling. → unblocks DSV4/GLM-5.1 token gen.
- **#2 MoE prepare/finalize audit** — rationale: identifies dispatch shape mismatches (Lightning Indexer vs router contract, 256E/8A invariants) that crash `MoEPrepareAndFinalizeNoDPEPModular` post-rebuild. → unblocks DSV4/GLM-5.1 token gen at TP=2×EP=8 target.
- **#3 MoE-DSA routing patch** — rationale: writes the coercion + adapter that resolves the audit's identified crash points, directly fixing the "MoE assert" that blocks `/v1/completions`. → unblocks DSV4/GLM-5.1 token gen.
- **#4 INT4-AWQ-Marlin sm_86 backport** — rationale: substitutes for `[128,128]` block-scaled FP8 in MoE expert path; without it DSV4 expert GEMM has no Ampere-native fast path. → unblocks DSV4/GLM-5.1 token gen at Silver-Gold tier.
- **#5 Per-expert AWQ calibration** — rationale: produces the INT4-AWQ checkpoint Story 4 needs; naive global calibration drops 256-expert MoE accuracy hard. → unblocks DSV4/GLM-5.1 token gen with ≤2% perplexity drop vs BF16 reference.
- **#6 TP=16 o_groups=8 relaxation (HIGH-RISK)** — rationale: fallback topology if EP=8 dispatch (Stories 2-3) blocks; keeps a token gen path open even if MoE-DSA patch drags. → unblocks DSV4 token gen via TP=16 if EP=8 path stalls.
- **#7 EPLB (stretch, post first coherent token)** — rationale: keeps 8 active experts hot under bursty traffic; needed for sustained Gold-tier aggregate (≥500 t/s) by reducing skew-induced expert stalls. → unblocks Gold-tier sustained token gen.
- **#8 MTP head (stretch, post first coherent token)** — rationale: DSV4-native 1.5-2× decode speedup; pushes conc=1 from Silver toward Diamond (75 t/s). → unblocks Platinum/Diamond conc=1 token gen.
- **#9 L1+L2 integration test for MoE-DSA dispatch** — rationale: regression gate for Stories 3-5; CC6's 1500-prompt correctness + perf gate depends on this exercising the 256E/8A path end-to-end. → unblocks safe iteration on token gen path without re-introducing crashes.

## Workflow

ship → commit → bridge milestone → next. L4 request at stories 3, 4, 9. Track A and Track B run in parallel — pick whichever next-story has cleanest path each time. **NEVER idle.** Pre-author all kernel/script logic ahead of L4 verdicts.

**Wall:** 15-25h focused. Track B is higher leverage but Track A unlocks EP=8 topology. Story 6 is HIGH-RISK isolated.

## Stories cut by CC0 prune

None — all 9 stories trace directly to DSV4 / GLM-5.1 token gen. Stretch demotions confirmed for #7 + #8 (already labeled, no demotion needed beyond explicit "post first coherent token" gate).
