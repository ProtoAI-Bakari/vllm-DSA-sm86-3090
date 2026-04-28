# 05 — Bench Results

> **Status: PENDING** — populated by CC8 (deploy + perf lane) after first L4 PASS.

When populated: dated tables from each L5 perf bench run including:

- Aggregate t/s at conc 1/2/4/8/16
- Per-GPU compute util %
- NCCL fabric saturation Gbps
- KV-cache utilization %
- Memory headroom margins
- Comparison vs llama.cpp PP baseline
- Tier classification (Bronze/Silver/Gold/Platinum/Diamond) at both conc=1 and aggregate scales

Raw bench data is in `bench/results/<date>/`. This doc is the human-readable summary.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
