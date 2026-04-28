# bench

Performance benchmark harness + results.

```
bench/
├── runners/        # CC8's bench scripts (concurrency sweep, NCCL profile, etc.)
├── golden/         # the canonical golden-numbers table
└── results/<date>/ # raw bench outputs, per-run
```

Raw results are gitignored (regenerable, large). The synthesized table lives in `docs/05_BENCH_RESULTS.md`.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
