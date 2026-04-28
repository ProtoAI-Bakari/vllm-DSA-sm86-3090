# bench/results — dated bench run artifacts

Each top-level dir = one bench run, named `<UTCdate>T<HHMMSS>Z__<profile>__<commit-short>/`.
The wrapper at `bench/run_full_bench.sh` writes here when `--out-base` lands inside the repo;
CC8's default `--out-base` is `~/AGENT/path_a_run_<ts>/` (outside repo, kept private until
correctness gate PASS), then a curated subset is mirrored here on PASS.

## Directory contract (per run)

```
<run-dir>/
├── 00_cordon.log                  # cordon_for_launch.sh + GID-index check
├── 02_gpu_util.csv                # per-GPU util at 100ms sampling
├── 03_kv_metrics.jsonl            # vLLM /metrics (KV usage, queue depths)
├── 04_nccl_profile/               # ibv_devinfo, topo, iperf3 mesh (56 pairs)
│   ├── ibv_cuda{1..8}.log
│   ├── topo_cuda{1..8}.log
│   ├── iperf3_<src>_to_<dst>.json
│   └── SUMMARY.txt
├── 05_conc_sweep.jsonl            # one line per conc step with raw-jsonl ref
├── 05_conc_sweep_conc{1,2,4,8,16,32}_raw.jsonl   # per-request results
├── 06_latency.json                # TTFT/TPOT/ITL/PP/TG p50/p95/p99
├── 07_tier.json                   # Bronze/Silver/Gold/Plat/Diamond classify
├── 08_optimize.json               # next-step candidates ranked by ROI
└── manifest.json                  # commit, profile path, endpoint, model, ts
```

## Acceptance gates required before mirroring here

The `bench/run_full_bench.sh` wrapper writes to `~/AGENT/path_a_run_<ts>/`
unconditionally. A run is mirrored to `bench/results/` only after:

- `correctness_replay.sh` PASS: top1 ≥0.98, cosine p95 ≥0.97, logprob_drift p95 ≤0.05
- CC6 1500-prompt fixed-seed gate PASS
- CC0 LEAD signs off via bridge `topic=l4_pass`

This keeps the public OSS repo's `bench/results/` clean — only verified runs.

## See also

- `docs/05_BENCH_RESULTS.md` — human-readable summary table
- `bench/profiles/*.yaml` — flight-deck profiles (TP/EP/KV/util knobs)
- `bench/runners/tier_classify.py` — tier ladder reference (corpus §8.1)
