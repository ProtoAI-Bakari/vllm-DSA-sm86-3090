# CC6 Story Backlog — Correctness Gate (1500-prompt + numerics)
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, xhigh effort, agent: CC6]**
**Last pruned:** 2026-04-28 04:08 PDT (per CC0 prune-to-project-goal-trace directive + 3-week master plan W1 scope)
**Branch:** `lane-cc6-gate` | **Project goal trace:** every story below ↓ unblocks DSV4/GLM-5.1 coherent token gen at perf tier.

Heaviest infra lane next to CC9. Build the gate harness so every CC3/CC4/CC5 ship can be validated automatically.

## Week 1 (T+0 → T+7d): FIRST COHERENT TOKEN — author + ship gate harness

| # | Story | Status | Output | Rationale (→ project goal) |
|---|---|---|---|---|
| 1 | 1500-prompt fixed-seed test set — curate from MMLU subset + DSV4-relevant + long-context | **DONE** (cba4390) | `tests/correctness/prompts.jsonl` | → without 1500-prompt set, regression detection has 15% noise floor (AO1) → CC6 PASS becomes meaningless → no merge gate → no integration confidence → no coherent token shipped |
| 2 | llama.cpp PP baseline capture script — conc=4-5 micro-batches against current GLM-5.1 PP head | **DONE** (cba4390) | `tests/correctness/baseline_capture.sh` + `capture_one.py` | → baseline.jsonl IS the ground truth that every rebuilt-vLLM under-test JSONL is graded against. No baseline → no comparison → no gate → no merge confidence |
| 3 | baseline.jsonl population — run capture (~30-60 min on conc=5) | **BLOCKED** endpoint | `tests/correctness/baseline.jsonl` | → see Story 2 rationale; mac4:8000 PP4 head dead 04:00, awaiting CC9 flush+restart of glm51_tp4 LaunchDaemon |
| 4 | top-1 token agreement gate — diff token sequences vs baseline | **DONE** (consolidated in verify_gate.py @ 035efac) | `tests/correctness/verify_gate.py` (top-1 logic) | → primary deterministic correctness signal; fails this → revert merge → DSV4/GLM-5.1 coherent tokens preserved |
| 5 | logit cosine gate — capture logits, compute cosine vs baseline | **DONE** (consolidated in verify_gate.py @ 035efac) | `tests/correctness/verify_gate.py` (per-position cosine) | → fine-grained numerics gate catches subtle degradation invisible to top-1 alone (e.g., low-prob token swaps) → catches before they manifest as coherence loss |
| 6 | perplexity gate — MMLU subset perplexity comparison | partial — needs real `cais/mmlu` corpus loader test | `tests/correctness/verify_gate.py` (mmlu_real branch) + MMLU loader | → ≤2% perplexity drop is the formal acceptance gate (LEAD §8.4); ensures rebuilt vLLM's coherence is statistically indistinguishable from baseline |
| 7 | L1 unit-test harness — per-kernel sm_86 numerics gate vs pytorch reference | **DONE** (55af8dd) | `tests/unit/{conftest,test_kernels_sm86.py,run_kernel_test.sh,gen_fixtures.py}` | → CC3/CC4/CC5 self-validate each kernel BEFORE integration → catches numerics bugs at unit level instead of L4 → cuts iteration cost from 25min/L4 to seconds/L1 → enables Week-1 first coherent token deadline |
| 11 | Hopper reference fallback — pure-pytorch CPU compute as ground truth for L1 | **DONE** (55af8dd) | `tests/correctness/cpu_reference.py` (5 kernel refs) | → without Hopper available on cluster, CPU pytorch IS the only ground truth for L1 → without L1 ground truth, CC3/CC4/CC5 ship blind → no first coherent token within W1 |

## Week 2 (T+7 → T+14d): MERGE-LOOP THROUGHPUT — daemonize + L2/L4

| # | Story | Status | Output | Rationale (→ project goal) |
|---|---|---|---|---|
| 8 | L2 forward-slice harness — load tiny test fixture, run kernel inside vLLM call chain on 1 node | next | `tests/integration/run_l2_forward_slice.sh` | → catches integration bugs (ABI mismatch, tensor layout, dispatcher routing) that L1 misses → without L2, integration bugs only surface in L4 (25 min/cycle) instead of L2 (≤2 min) → too-slow to hit W2 throughput target of 30-50 L4 cycles |
| 9 | L4 integration verdict — given a CC9 capture, run all 3 numerics gates + grill_33 + post bridge l4_result | next | `tests/correctness/run_l4_verdict.sh` | → CC9 needs a single-command verdict runner so each merge candidate either ships (PASS → CC0 merges) or reverts (FAIL → CC0 reverts); without it CC0 is on the hook to manually compute → bottleneck → throughput collapse |
| 10 | Report generator — per-integration markdown report with diff tables + cosine plot + perplexity delta | **DONE** (consolidated in verify_gate.py render_markdown @ 035efac) | `tests/correctness/verify_gate.py` (auto-emits to `~/AGENT/comms/CC6_GATE_RESULTS_<integ>.md`) | → human-auditable trail: when z asks "why did CC4 rev2 ship and rev3 revert", the report answers immediately → without it CC0 has to dig through raw JSONL on every dispute |
| 12 (W2 NEW) | L4 verdict daemon — agent_lifecycle.sh-driven autopoll for CC9 captures | next | `tests/correctness/l4_daemon.sh` | → master plan §"daemon roles": gate must run continuous so CC9 ships and gets verdict back without me being prompted → throughput unblocker for W2 30-50 cycles target |
| 13 (W2 NEW) | L3 small-model load — single-rank tinyllama load + 1-prompt forward through rebuilt _C.abi3.so | next | `tests/integration/run_l3_small_model.sh` | → bridges L2 (kernel-in-call-chain) and L4 (full GLM-5.1 PP) → catches python-side wiring + module-load bugs at single-node scale before paying the 270GB-load + 16-rank cost of L4 |

## Week 3 (T+14 → T+21d): GOLD PUSH — full perplexity, regression, post-merge automation

| # | Story | Status | Output | Rationale (→ project goal) |
|---|---|---|---|---|
| 14 (W3 NEW) | Full perplexity eval — entire `cais/mmlu` test set (14K Qs across 57 subjects) | next | `tests/correctness/perplexity_full.py` | → W3 acceptance is Gold tier; tier requires confidence baseline-vs-rebuilt is statistically indistinguishable across the FULL MMLU corpus, not just 300-prompt subset → without it Gold claim is unsubstantiated |
| 15 (W3 NEW) | Long-context regression suite — 8K/16K/32K context prompts + KV-cache stress | next | `tests/correctness/long_ctx_regression.sh` | → DSV4 + GLM-5.1 are MoE long-context models; W3 stretch optimizations (EPLB, MTP, INT8 KV) all interact with KV cache → must validate they don't degrade long-context coherence |
| 16 (W3 NEW) | Automated grill_33 post-merge — adversarial 33-prompt set fired immediately after CC0 merge | next | `tests/correctness/grill_33.sh` + bridge l4_result hook | → catches "regression that hides in average but explodes on edge cases" → W3 throughput needs grill auto-fired so CC0 doesn't manually invoke after every merge |

---

## CUTS / DEMOTIONS (per CC0 prune directive 2026-04-28)
None for CC6 — every story above traces directly to "DSV4 + GLM-5.1 coherent tokens at perf tier". No story cut.

## Workflow rules (master plan §Daily cadence)
- Stories 3 (capture) is BACKGROUND once endpoint live — own that buffer; do other stories during.
- Stories 4-6 + 10 are CONSOLIDATED inside `verify_gate.py` (single-runner cuts L4 verdict surface area).
- Story 9 (L4 runner) requires CC9 to ship a capture — request via bridge `topic=l4_capture_ready` poll.
- Between stories: KEEP AUTHORING. Never block waiting for L4 verdict (async per master plan).
- Per-30-min: bridge milestone + status to `/Users/z/AGENTIC/comms/CC6_STATUS.md`.

## Wall budget
- W1: ~16h focused (already on track: ~75% complete except Story 3 endpoint-blocked).
- W2: ~24h (L2 + L4 daemon + L3 + report polish).
- W3: ~32h (full perplexity + long-ctx + grill auto).
- Total: 80-120h matches master plan §Per-agent table.
