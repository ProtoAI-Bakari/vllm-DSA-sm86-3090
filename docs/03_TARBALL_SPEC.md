# Tarball Specification — `vllm-dsa-sm86_v0.20.0+cu128_amd64.tar.gz`
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]**

<!-- --ProtoAI-Bakari-- -->

## Why a venv tarball (not .deb / wheel-only)

vLLM 0.20.0 plus the sm_86 DSA rebuild produces a compiled `vllm/_C.abi3.so` whose ABI is pinned to a specific `(torch, cuda, glibc)` triple. Three options were considered:

| Option | Pro | Con | Verdict |
|---|---|---|---|
| `.deb` package | apt-managed, dependency graph | vLLM has no upstream deb workflow; we'd own a debian/ tree forever | **rejected** |
| Wheel (`.whl`) only | pip-installable | doesn't carry torch+cuda+system libs; ABI breakage on node drift | rejected as primary; published as artifact |
| **venv tarball** | self-contained, atomic, rollback-safe | larger (~3-5 GB) | **accepted** |

Tarball wins because every node in ZCCX runs the same Ubuntu 22.04 + CUDA 12.8 + driver line, and an atomic `extract → activate → smoke` flow gives clean rollback if any node fails verification.

## Filename contract

```
vllm-dsa-sm86_v0.20.0+cu128_amd64.tar.gz
```

| Field | Value | Source of truth |
|---|---|---|
| `vllm-dsa-sm86` | package name | this repo |
| `v0.20.0` | upstream vLLM tag we patched on | `pyproject.toml` `[project] version` |
| `+cu128` | CUDA build tag | `nvcc --version` at build time |
| `_amd64` | architecture | `dpkg --print-architecture` |
| `.tar.gz` | tar + gzip | reproducible: `tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2026-01-01' -czf …` |

**Build SHA stamp** lives inside the tarball at `BUILD_INFO.json` (next section), NOT in the filename — keeps the filename short for fanout scripts.

## Tarball layout (extracted)

```
vllm-dsa-sm86/
├── BUILD_INFO.json                 # provenance: git SHA, builder, build time, GPU arch list, hashes
├── INSTALL.sh                      # idempotent installer (consumed by scripts/install_cluster.sh)
├── UNINSTALL.sh                    # restore prior venv from /home/z/.venvs/.venv-vllm-dsa-sm86.prev
├── activate                        # convenience shim — sources venv/bin/activate
├── venv/                           # full python venv tree (relocatable via `--copies`)
│   ├── bin/
│   │   ├── python3.12              # pinned interpreter
│   │   ├── pip
│   │   ├── vllm                    # vLLM CLI (entrypoint)
│   │   └── vllm-serve              # our wrapper (next section)
│   ├── lib/python3.12/site-packages/
│   │   ├── vllm/
│   │   │   ├── _C.abi3.so          # **the rebuilt sm_86 .so**
│   │   │   ├── _moe_C.abi3.so
│   │   │   ├── _flashinfer_C.abi3.so
│   │   │   └── …                    # rest of vllm tree
│   │   ├── torch/                  # 2.5.x +cu128 wheel
│   │   ├── triton/
│   │   ├── flash_attn/             # if bundled
│   │   └── …                        # transitive deps frozen via pip freeze
│   └── pyvenv.cfg                  # patched by INSTALL.sh to point at /home/z/.venvs/.venv-vllm-dsa-sm86
├── patches/                        # the 17-patch cascade (.sh files) — for audit + reproducibility
│   └── README.md → ../patches/01_dsa_sm86_kernels/
├── share/
│   ├── examples/
│   │   ├── glm51_tp2_ep8.yaml      # reference SkyPilot launch
│   │   └── dsv4_tp2_ep8.yaml
│   └── doc/
│       ├── README.md               # short user-facing readme inside tarball
│       └── LICENSE                 # Apache-2.0
└── checksums.sha256                # `sha256sum venv/lib/python3.12/site-packages/vllm/*.so` etc.
```

**Total uncompressed size budget:** 3.5–4.5 GiB. Compressed target: ≤ 1.6 GiB (gzip -9). If we exceed 1.8 GiB compressed, switch to zstd `-19 --long=27` and update filename to `.tar.zst`.

## `BUILD_INFO.json` schema

```json
{
  "package": "vllm-dsa-sm86",
  "version": "0.20.0+cu128",
  "git_sha": "<full 40-char sha>",
  "git_remote": "https://github.com/ProtoAI-Bakari/vllm-DSA-sm86-3090",
  "git_branch": "lane-cc7-package",
  "built_at": "2026-04-28T03:32:00Z",
  "built_by": "claude-cc7",
  "builder_host": "cuda7",
  "torch_version": "2.5.1+cu128",
  "cuda_version": "12.8",
  "cuda_arch_list": ["8.6"],
  "python_version": "3.12.x",
  "ubuntu_version": "22.04",
  "glibc_version": "2.35",
  "patches_applied": [
    "patch_load_w13_diag",
    "patch_dsv4_moe_block_v2_revert",
    "patch_dsv4_attn_groups",
    "patch_dsv4_supports_pp",
    "patch_dsv4_pp_skip_nonlocal",
    "patch_dsv4_pp_skip_v2",
    "patch_dsv4_pp_skip_v3",
    "patch_pp_kv_skip_indexer",
    "patch_dsv4_swa_skip_v1",
    "patch_dsv4_swa_skip_v2",
    "patch_skip_finfer_warmup",
    "patch_dsv4_dummy_short_circuit",
    "patch_compressor_state_skip",
    "patch_sparse_indexer_stub"
  ],
  "kernels_rebuilt": [
    "sparse_attn_indexer",
    "lightning_compressor",
    "mla_sparse",
    "swa",
    "moe_dispatch"
  ],
  "so_hashes": {
    "vllm/_C.abi3.so": "<sha256>",
    "vllm/_moe_C.abi3.so": "<sha256>",
    "vllm/_flashinfer_C.abi3.so": "<sha256>"
  },
  "correctness_gate": {
    "passed": true,
    "cc6_report": "/Users/z/AGENT/comms/CC6_GATE_REPORT_<timestamp>.md",
    "prompts_evaluated": 1500,
    "top1_match": 0.98,
    "cosine": 0.97,
    "perplexity_delta_pct": 1.6
  }
}
```

`BUILD_INFO.json` is single source of truth for "is this tarball legit". `verify_install.sh` (Story 3) reads it post-extract and refuses to activate if `correctness_gate.passed != true`.

## `vllm-serve` wrapper (cmd-line entrypoint)

Thin wrapper over `python -m vllm.entrypoints.openai.api_server` that:

1. Sources the venv (no manual `activate` needed)
2. Sets `CUDA_VISIBLE_DEVICES`, `VLLM_USE_FLASHINFER=1`, `TORCH_CUDA_ARCH_LIST=8.6` defaults
3. Logs `BUILD_INFO.json` to stderr at startup so traces always show which build served the request
4. Defers all other args to vLLM (`exec python -m vllm.entrypoints.openai.api_server "$@"`)

Lives at `venv/bin/vllm-serve`. Symlinked into `/usr/local/bin/vllm-serve` by `INSTALL.sh` (only if `/usr/local/bin` is writable — fanout uses sudo via approval gate).

## `INSTALL.sh` contract (consumed by Story 2 fanout)

```
INSTALL.sh [--target /home/z/.venvs/.venv-vllm-dsa-sm86] [--prev-snapshot] [--cluster-vetted]
```

Behavior:
1. Verify `BUILD_INFO.json` exists + correctness_gate.passed
2. If `--prev-snapshot`: `mv $TARGET $TARGET.prev` (atomic rename)
3. `cp -a venv/ $TARGET/`
4. Re-write `$TARGET/pyvenv.cfg` paths to absolute target
5. `$TARGET/bin/python -c "import vllm; print(vllm.__version__)"` — fail loudly if it doesn't return our patched version string
6. Print SUCCESS line: `INSTALL OK: vllm-dsa-sm86 v0.20.0+cu128 sha=<short-sha> target=$TARGET`

Exit codes: 0 success, 10 BUILD_INFO missing, 11 correctness_gate fail, 12 venv copy fail, 13 import smoke fail.

`--cluster-vetted` flag tells `block_unapproved_cluster_op_hook.py` this fanout was approved through `approval_gate.py` — set ONLY by `scripts/install_cluster.sh` after the approval queue returns APPROVED for the fanout request.

## `UNINSTALL.sh` contract (rollback)

```
UNINSTALL.sh [--target /home/z/.venvs/.venv-vllm-dsa-sm86]
```

1. Refuse if `$TARGET.prev` does not exist (no-op rather than data loss)
2. `rm -rf $TARGET`
3. `mv $TARGET.prev $TARGET`
4. Smoke: `$TARGET/bin/python -c "import vllm; print(vllm.__version__)"` — confirms rollback restored a working venv
5. Emits `ROLLBACK OK: target=$TARGET restored_version=<x>`

`scripts/rollback_install.sh` (Story 4) wraps `UNINSTALL.sh` across all 8 nodes when verify_install.sh reports any node-level failure.

## Reproducible build constraint

The tarball MUST be byte-reproducible from a given (git SHA, torch wheel, CUDA toolkit) triple. Achieved by:
- `tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2026-01-01'`
- `gzip -n -9` (no timestamp in gzip header)
- `pip install --no-build-isolation` against pinned `requirements.lock` committed at repo root
- `__pycache__/` stripped before packaging (`find venv -name __pycache__ -type d -prune -exec rm -rf {} +`)
- `.dist-info/RECORD` files have stable hashes (achieved by stripping then regenerating)

`scripts/build_multiarch.sh` (Story 8 stretch) extends this to `+sm80,+sm86,+sm89` flavors using the same envelope.

## Acceptance check (CC7 gate)

Tarball is shippable when:
1. Filename matches contract above
2. `tar -tzf $TARBALL | head` shows `vllm-dsa-sm86/BUILD_INFO.json` first
3. `BUILD_INFO.json` validates against this schema (jsonschema check — Story 7)
4. `INSTALL.sh` on a fresh ZCCX node reports SUCCESS in <60s
5. Post-install `python -c "import vllm; print(vllm.__version__)"` returns the patched version string with `+sm86` suffix

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC7]
