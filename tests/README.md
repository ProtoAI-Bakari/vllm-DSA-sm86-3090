# tests

Three tiers of tests — see `docs/04_TEST_PYRAMID.md` for full description.

```
tests/
├── unit/           # L1 — single 3090, kernel bit-correctness
├── integration/    # L2/L3 — 1-2 nodes, vLLM-attached forward slices
└── correctness/    # L4 — full cluster, real model token gen
```

CC6 owns the harness; community contributors run whatever tier their hardware supports.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
