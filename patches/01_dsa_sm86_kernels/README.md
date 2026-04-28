# patches/01_dsa_sm86_kernels

NEW kernel ports for DSA chain on sm_86. CC3 / CC4 / CC5 commit here.

Each port is one directory:
- `sparse_attn_indexer/` — CC3
- `compressor/` — CC4
- `mla_sparse/` — CC4
- `swa/` — CC4
- `moe_dispatch_dsa/` — CC5

Per-port files:
- `<kernel>_sm86.cu` — the actual ported kernel source
- `<kernel>_sm86.h` — header
- `<kernel>_unit_test.cu` — L1 unit test (CC6 references)
- `<kernel>_AUDIT.md` — kernel-by-kernel audit trail
- `<kernel>_NUMERICS.json` — numerics retention from last L4 run

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
