# 05 — Bench Results

> **Status: PENDING** — populated by CC8 (deploy + perf lane) after first L4 PASS.
>
> CC8 pre-author: harness ready (commits fa74b4d / accc201 / 0a4630f). Awaits CC7 cluster fanout + CC6 correctness gate PASS.

This doc is the human-readable summary of every L5 perf bench run on the
ZCCX 8x2 cluster. Raw artifact directories sit at `bench/results/<date>/`
(or `~/AGENT/path_a_run_<ts>/` while the public `bench/results/` mirror is
empty). The published numerics retention table (per `CONTRIBUTING.md` L5 tier)
is the bottom of this doc.

## Pipeline (one-shot)

```
bench/run_full_bench.sh \
  --endpoint http://cuda1:8000 \
  --model glm51-iq2xxs \
  --profile bench/profiles/glm51_tp2_ep8.yaml \
  --concs 1,2,4,8,16,32 \
  --requests 64 \
  --latency-conc 8
```

Composes stages: cordon → endpoint sentinel → gpu_util_capture (bg) →
kv_metrics (bg) → nccl_profile (one-shot) → conc_sweep → latency_percentile →
tier_classify → optimize_loop → write `PATH_A_FINAL_RESULTS_<datetime>.md`.

## Schema (per run)

Every bench run emits the following columns; CC6's correctness gate gates
inclusion in this published doc.

| Column | Source | Notes |
|---|---|---|
| date | run_ts | UTC `YYYY-MM-DDTHH:MM:SSZ` |
| profile | bench/profiles/*.yaml | flight-deck profile name |
| commit | git rev-parse HEAD | lane-cc8-deploy SHA |
| conc | conc_sweep.jsonl | 1, 2, 4, 8, 16, 32 |
| agg_tg_tps | conc_sweep + latency | total completion tokens / wall |
| per_req_tg_tps_mean | conc_sweep | mean per-request decode rate |
| ttft_ms | latency_percentile p50/p95/p99 | streaming first-token wall |
| tpot_ms | latency_percentile p50/p95/p99 | (wall - ttft) / (n_completion - 1) |
| itl_ms | latency_percentile p50/p95/p99 | inter-token deltas (excludes prefill) |
| pp_tps_mean | latency_percentile | n_prompt / ttft |
| gpu_util_mean_% | gpu_util_capture.csv | nvidia-smi 100ms samples |
| kv_usage_mean_% | kv_metrics.jsonl | vllm:gpu_cache_usage_perc |
| nccl_p95_Gbps | nccl_profile/iperf3_*.json | mesh p95 over 56 pairs |
| top1_vs_pp | CC6 acceptance gate | ≥0.98 required |
| ppl_drift | CC6 acceptance gate | ≤2% MMLU subset required |
| tier_aggregate | tier_classify | Bronze/Silver/Gold/Platinum/Diamond/Cosmic |
| tier_conc1 | tier_classify | same scale, conc=1 ladder |

## Tier ladder (corpus §8.1)

| Tier | Aggregate t/s | Conc=1 t/s |
|---|---|---|
| Bronze | ≥ 100 | 25 |
| Silver | ≥ 300 | 35 |
| Gold | ≥ 500 | 55 |
| Platinum | ≥ 700 | 65 |
| Diamond | ≥ 900 | 75 |
| Cosmic | ≥ 1000 | n/a |

## Numerics retention (public reference outputs)

For each profile + commit, the published reference set:

- 64 fixed-seed prompts (deterministic, seed=42, see `tests/correctness/baseline_capture.sh`)
- temperature = 0
- max_tokens = 256
- model + tokenizer revision pinned by HF SHA

L0/L1 contributors with a single 3090 can run the same prompts against their
local rebuild and compare top-1 / cosine / perplexity to these reference
outputs. CC6 publishes the reference once correctness gate first passes.

## Results

> _empty until first L4 PASS._

| date | profile | commit | conc | agg_tg_tps | conc1_tg_tps | ttft_p50 | tpot_p50 | gpu_util | kv_usage | nccl_p95 | tier_agg | tier_conc1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Below-Bronze ceiling (transparent)

Per AO1 audit + corpus §8.2, **DSV4 BF16-emulation on Ampere is physics-bound at
10-20 t/s aggregate**. Any DSV4 entry at below-Bronze in this table is
expected on the BF16 path — INT4-AWQ-Marlin (CC5 lane) is the unblocking
intervention for the silver-gold target. Do NOT confuse this ceiling with a
broken kernel.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC8]
