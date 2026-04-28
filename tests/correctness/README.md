# tests/correctness — CC6 Gate (full toolchain)

The 1500-prompt fixed-seed correctness gate that runs after every CC3/CC4/CC5
integration into the rebuilt `vllm/_C.abi3.so`. CC0 reverts any merge that
fails it.

**Gate criteria** (LEAD corpus PART2 §8.4):

| Metric                       | Threshold | Notes                                                  |
|------------------------------|-----------|--------------------------------------------------------|
| top-1 text match             | ≥ 98 %    | deterministic temp=0 → exact match expected            |
| logit cosine                 | ≥ 0.97    | requires `logprobs` on both baseline + under-test      |
| MMLU perplexity drop         | ≤ 2 %     | requires `mmlu_real` category + logprobs both sides    |
| word Jaccard (fallback)      | ≥ 0.95    | used when logprobs missing                             |
| Long-context needle recall   | ≥ 90 %    | 8K/16K/32K needle-in-haystack (Story 15)               |
| Tier (LEAD §8.1)             | Bronze+   | aggregate ≥ 100 t/s OR conc=1 ≥ 25 t/s                 |

## Files

| File                          | Purpose                                                                     |
|-------------------------------|-----------------------------------------------------------------------------|
| `gen_prompts.py`              | Story 1 — deterministic seed=42 generator → `prompts.jsonl` (1500 lines).   |
| `prompts.jsonl`               | Generated; 9 categories balanced (factual 200 / mmlu 300 / reasoning 200 / code 200 / summ 150 / creative 150 / multilingual 100 / repetition 100 / edge 100). |
| `capture_one.py`              | Story 2 — single-prompt SSE-streaming POST; TTFT/TPOT/ITL/PP_TPS/TG_TPS metrics. |
| `baseline_capture.sh`         | Story 2 — resumable, conc=5 default via xargs -P, env-configurable.         |
| `verify_gate.py`              | Stories 4+5+6+10 (consolidated) — top-1 + cosine + perplexity + Jaccard + per-category report. |
| `cpu_reference.py`            | Story 11 — pytorch CPU references for 5 sm_86 kernels.                      |
| `run_l4_verdict.sh`           | Story 9 — single-command L4 grade runner; bridges `topic=l4_result`.        |
| `l4_daemon.sh`                | Story 12 — polls bridge `l4_capture_ready`; dispatches verdict runs.        |
| `grill_33.sh` + `grill_33_gen.py` | Story 16 — 33-prompt adversarial post-merge auto-fire.                  |
| `perplexity_full.py`          | Story 14 — full MMLU per-subject accuracy + correct-letter logprob drop.    |
| `gen_long_ctx_prompts.py`     | Story 15 — 8K/16K/32K needle-in-haystack prompt generator.                  |
| `long_ctx_regression.sh` + `score_long_ctx.py` | Story 15 — long-context capture + score.                   |
| `endpoint_health_daemon.sh`   | W3-NEW — polls PP endpoint, posts bridge `endpoint_health` UP/DOWN, optional auto-fire baseline. |
| `profile_to_jsonl.py`         | W3-NEW — converts CC9 profile JSON → gate JSONL.                            |
| `tier_classify.py`            | W3-NEW — CC8 bench JSON → LEAD-tier verdict (aggregate + conc=1 axes).      |
| `ship_l4_capture.sh`          | W3-NEW — single CC9 entrypoint: detect format, grade, optionally chain grill+longctx. |
| `test_gate_self.sh`           | W3-NEW — 5 synthetic positive/negative self-tests for the gate runner.      |
| `coverage_matrix.py`          | W3-NEW — kernel × test-tier coverage audit; emits action items for CC3/CC4/CC5. |
| `cc6_daily_status.py`         | W3-NEW — bridge.db digest for CC0's 3-min cron.                             |

## Workflow — typical CC9 integration cycle

```
# 1. CC2/CC3/CC4/CC5 ship a candidate _C.abi3.so via cc_git.
# 2. CC9 captures a profile dump from cluster smoke.
# 3. CC9 invokes the gate via single entrypoint:

bash tests/correctness/ship_l4_capture.sh \
    --integ cc4-mla-rev3 \
    --capture /repo/models/RUN/PROFILE_RESULTS/<...>.json \
    --baseline tests/correctness/baseline.jsonl \
    --grill                    # optional — fires grill_33 too
    --longctx                  # optional — fires long-ctx regression

# 4. Bridge posts l4_shipped <PASS|FAIL>; CC0 merges on PASS or reverts on FAIL.
```

## Workflow — daemon mode (continuous, post-master-plan)

```
# CC6 gate daemon (listens for new captures):
bash tests/correctness/l4_daemon.sh                  &  # poll every 30s

# CC6 endpoint daemon (auto-detects PVLM endpoint state changes):
AUTO_FIRE_BASELINE=1 \
  bash tests/correctness/endpoint_health_daemon.sh   &  # poll every 60s
```

## Self-tests (run anytime)

```
bash tests/correctness/test_gate_self.sh        # 5 positive+negative gate fixtures
bash tests/unit/run_kernel_test.sh              # L1 numerics vs pytorch ref
bash tests/integration/run_l2_forward_slice.sh  # L2 kernel-in-call-chain
bash tests/integration/run_l3_small_model.sh    # L3 single-rank tinyllama load
python3 tests/correctness/coverage_matrix.py    # coverage audit
```

## Daily status

```
python3 tests/correctness/cc6_daily_status.py --hours 24
# → ~/AGENT/comms/CC6_DAILY_STATUS.md  (and bridge daily_status)
```

## Categories in `prompts.jsonl`

| category           | n   | max_tok | purpose                                            |
|--------------------|----:|--------:|----------------------------------------------------|
| factual            | 200 | 16      | short-answer, low-entropy ideal for top-1          |
| mmlu_placeholder   | 300 | 4       | MCQ schema; replaced by `mmlu_real` if `MMLU_DIR` set |
| reasoning          | 200 | 96      | multi-step word problems                           |
| code               | 200 | 160     | short Python/Bash completions                      |
| summarization      | 150 | 96      | paragraph → 2-sentence summary                     |
| creative           | 150 | 64      | style-constrained 1-2 sentence generation          |
| multilingual       | 100 | 48      | translation                                        |
| repetition         | 100 | 80      | attention-sink adversarial                         |
| edge               | 100 | 64      | empty / 1-tok / special chars / very-long          |
| **total**          |**1500**|         |                                                  |

## Per-integration reports

Reports go to `/Users/z/AGENT/comms/`:
- `CC6_GATE_RESULTS_<integ>.md`     — primary numerics report (top-1 / cosine / perplexity / Jaccard / per-category)
- `CC6_GRILL33_<integ>.md`          — adversarial 33-prompt result
- `CC6_LONGCTX_<integ>.md`          — long-context needle recall result
- `CC6_PERPLEXITY_<integ>.md`       — full MMLU per-subject accuracy + log-prob drop
- `CC6_TIER_<integ>.md`             — LEAD-tier verdict
- `CC6_COVERAGE_MATRIX.md`          — kernel × test-tier coverage audit
- `CC6_DAILY_STATUS.md`             — last-24h CC6 bridge digest
