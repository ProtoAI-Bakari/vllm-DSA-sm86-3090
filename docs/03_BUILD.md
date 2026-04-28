# 03 — Build Instructions
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agents: CC2 + CC7]**

<!-- --ProtoAI-Bakari-- -->

This document covers two audiences:
- **Maintainers** (CC2 + CC7): reproducing the official tarball byte-for-byte from a tagged git SHA.
- **Contributors** (L0/L1): rebuilding on a single 3090 to test changes without ZCCX.

The tarball spec (`docs/03_TARBALL_SPEC.md`) is the authoritative description of *what* gets built. This doc is the authoritative description of *how*.

---

## 0. Prerequisites

| Tool | Pinned version | Source of truth |
|---|---|---|
| Ubuntu | 22.04 | `lsb_release -a` |
| glibc | 2.35 | `ldd --version` |
| Python | 3.12.x | `pyproject.toml` |
| CUDA Toolkit | 12.8 | `nvcc --version` |
| NVIDIA driver | ≥ 550 | `nvidia-smi` |
| GCC | 11.4 | `gcc --version` |
| cmake | ≥ 3.27 | `cmake --version` |
| ninja | ≥ 1.11 | `ninja --version` |
| ccache | ≥ 4.8 | `ccache --version` |
| torch | 2.5.x +cu128 | `requirements.lock` |

A repeatable rebuild requires this exact stack — drift in any row produces a tarball that fails byte-equality vs upstream releases. Contributors targeting personal builds may relax CUDA/driver but **must NOT** publish artifacts under the official filename.

---

## 1. Quick path — rebuild on a fresh ZCCX-class box (maintainer)

```bash
# 1. Clone at the tag you want to ship
git clone https://github.com/ProtoAI-Bakari/vllm-DSA-sm86-3090.git
cd vllm-DSA-sm86-3090
git checkout v0.20.0+cu128         # release tag (set by CC0 on merge)

# 2. Toolchain check
./scripts/build_env.lock           # asserts every row in §0
                                   # (CC2 owns; refuses if any row drifts)

# 3. Build the .so
TORCH_CUDA_ARCH_LIST="8.6" \
CCACHE_DIR=$HOME/.ccache/vllm-dsa-sm86 \
bash ./scripts/build_vllm_dsa_sm86.sh \
  --jobs $(nproc) \
  --output ./build/vllm-dsa-sm86

# 4. Pack the tarball (Story 7 covers verification of the output)
bash ./scripts/pack_tarball.sh \
  --input  ./build/vllm-dsa-sm86 \
  --output /repo/INSTALLERS/vllm-dsa-sm86_v0.20.0+cu128_amd64.tar.gz

# 5. Smoke
bash ./tests/correctness/tarball_smoke.sh /repo/INSTALLERS/vllm-dsa-sm86_v0.20.0+cu128_amd64.tar.gz
```

---

## 2. From-source rebuild details (CC2 lane)

CC2 owns this section. Steps 2.1–2.5 are what `scripts/build_vllm_dsa_sm86.sh` executes.

### 2.1 Clone vLLM at pinned upstream commit
```
VLLM_PINNED_SHA=<set by CC2 in scripts/build_env.lock>
git clone https://github.com/vllm-project/vllm.git ./build/vllm
git -C ./build/vllm checkout "$VLLM_PINNED_SHA"
```

### 2.2 Apply the 17-patch cascade + sm_86 kernel patches
```
for p in patches/00_baseline_p6_p7_p8/*.sh patches/01_dsa_sm86_kernels/*.sh; do
  bash "$p" ./build/vllm
done
```
The sm_86 kernel patches replace TMA → cp.async.cg, WGMMA → mma.sync.aligned.m16n8k16, cp.async.bulk → cp.async, and PDL → host sync. See `docs/01_ARCHITECTURE.md` for the kernel-level mapping.

### 2.3 cmake configure
```
cd ./build/vllm
cmake -B build \
  -DTORCH_CUDA_ARCH_LIST="8.6" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVLLM_USE_PRECOMPILED=0 \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -GNinja
```

### 2.4 Build
```
cmake --build build --parallel "$(nproc)"
```
First-build wall: ~25–40 min on a 16-core box with cold ccache. Subsequent rebuilds: 2–5 min if only kernel sources changed.

### 2.5 ABI lock + smoke
```
python -c "import vllm._C as c; print(c.__file__)"
ldd build/vllm/_C.abi3.so | grep -E 'libtorch|libcuda'    # confirm pinned wheel
```
Mismatched libtorch / libcuda paths → the tarball will fail on cuda<n> nodes whose driver was never updated. CC2's `scripts/verify_env.sh` enforces this before pack.

---

## 3. Tarball pack (CC7 lane)

`scripts/pack_tarball.sh` (CC7-owned):
1. Copies the built tree into a clean staging dir under `vllm-dsa-sm86/`.
2. Strips `__pycache__`, `*.pyc`, build cache, test artifacts.
3. Writes `BUILD_INFO.json` from the running env (git SHA, torch, cuda, ccache stats, gate result).
4. Computes per-`.so` sha256 → `checksums.sha256`.
5. Tars with reproducible flags:
   ```
   tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2026-01-01' \
       -cf - vllm-dsa-sm86/ | gzip -n -9 > $OUTPUT
   ```
6. Asserts the output sha256 matches the prior release's checksum **only when** rebuilding the same git SHA on the same toolchain — divergence flags non-reproducibility (CC2 + CC7 investigate).

---

## 4. Single-3090 contributor path (L0/L1)

Aimed at OSS contributors who don't have ZCCX. Goal: build the wheel, run unit tests on a single RTX 3090, push a PR. **Not** a path to ship release artifacts.

```
# 1. Set up a Python 3.12 venv against torch 2.5 +cu128
python3.12 -m venv ~/.venv-vllm-dev
source ~/.venv-vllm-dev/bin/activate
pip install -r requirements.lock

# 2. Do the same build from §2 on your local machine
git clone https://github.com/ProtoAI-Bakari/vllm-DSA-sm86-3090.git
cd vllm-DSA-sm86-3090
TORCH_CUDA_ARCH_LIST="8.6" bash ./scripts/build_vllm_dsa_sm86.sh

# 3. Run L0 + L1 tests (no model load — pytorch reference fixtures only)
pytest tests/unit -v               # CC4/CC6-owned kernel L1 tests
bash tests/unit/run_kernel_test.sh # standalone harness

# 4. PR the patched kernel
git checkout -b your-feature
# edit csrc_patches/...
git commit -m "your change"        # .githooks/prepare-commit-msg adds Assisted-By:
git push origin your-feature
```

The L0/L1 path only validates kernel-level numerics — it cannot validate end-to-end DSA serving. End-to-end (L4) requires the ZCCX cluster and is gated by `approval_gate.py`.

---

## 5. Deterministic-build escape hatches

If your rebuild's tarball sha256 disagrees with the released artifact, the typical culprits in priority order:

1. **`__pycache__`** survived the strip → bytecode timestamps differ. Fix: `find . -name __pycache__ -prune -exec rm -rf {} +`.
2. **`.dist-info/RECORD`** files have hash drift due to pip metadata regen. Fix: `pip install --no-build-isolation` plus `RECORD` regeneration script in `pack_tarball.sh`.
3. **`gzip` header timestamp** snuck in. Fix: ensure `gzip -n` (no name/timestamp).
4. **`tar` ordering** changed. Fix: `--sort=name`.
5. **`ccache` injected paths** into compiled `.so` `__FILE__` macros. Fix: `CCACHE_BASEDIR=$PWD CCACHE_NOHASHDIR=1`.
6. **CUDA toolkit minor version drift** (e.g. 12.8.0 → 12.8.1). Captured in `BUILD_INFO.cuda_version`; if drift is intentional, bump release tag and re-cut.

---

## 6. Cross-version matrix (Week 3 stretch — CC2)

Eventual support: vLLM 0.19.0 + 0.20.0 dual-track. Each version produces a separate tarball with version baked into both filename + BUILD_INFO. `scripts/build_vllm_dsa_sm86.sh --vllm-version 0.19.0` selects the upstream pin. Test matrix lives at `tests/correctness/matrix.yml` (CC8-owned — not yet authored).

---

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agents: CC2 + CC7]
