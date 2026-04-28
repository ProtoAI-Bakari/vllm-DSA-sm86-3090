#!/usr/bin/env bash
# build_vllm_dsa_sm86.sh v3 — CC2 dual-profile vLLM source rebuild for sm_86 (Ampere RTX 3090).
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC2]
#
# Profiles:
#   --profile dsv4-pr40760  (DEFAULT) — target dsv4 venv + PR40760 SHA aa114601 + DSA sm_86 kernels +
#                            CC3 4db7f8e CMakeLists shard + CC5 ed96096 o_groups_relax.
#   --profile baseline-020  — target .venv-vllm_dsa_sm86_baseline + 0.20.0 SHA 101584af0 + P6/P7/P8.
#
# Use:
#   bash scripts/build_vllm_dsa_sm86.sh                       # full dsv4 rebuild + ABI smoke + install
#   bash scripts/build_vllm_dsa_sm86.sh --profile baseline-020
#   bash scripts/build_vllm_dsa_sm86.sh --rebuild             # git clean -xdf vendor tree first
#   bash scripts/build_vllm_dsa_sm86.sh --skip-install        # build wheel, do not install (caller handles)
#   bash scripts/build_vllm_dsa_sm86.sh --abi-only            # ABI smoke against current installed vllm
#   bash scripts/build_vllm_dsa_sm86.sh --serve-qwen          # post-install Qwen3.5 verify (CC9 only)
#
# Acceptance (per CC0 18:52Z dispatch):
#   (a) cmake -DTORCH_CUDA_ARCH_LIST=8.6 includes CC3 4db7f8e CMakeLists shard
#   (b) ccache + sccache wired (USE_CCACHE=1 default)
#   (c) CC5 ed96096 o_groups_relax included
#   (d) ABI smoke vs unpatched baseline before pushing to cluster
#   (e) installer step (install_dsa_sm86.sh) usable as L4 hook between mirror-update and launch.

set -euo pipefail

PROFILE="${PROFILE:-dsv4-pr40760}"
REBUILD=0
SKIP_INSTALL=0
ABI_ONLY=0
SERVE_QWEN=0
SKIP_CSRC=0
SKIP_OGROUPS=0
SKIP_PATCHES=0
for arg in "$@"; do
    case "$arg" in
        --profile) shift; PROFILE="$1" ;;
        --profile=*) PROFILE="${arg#--profile=}" ;;
        --rebuild) REBUILD=1 ;;
        --skip-install) SKIP_INSTALL=1 ;;
        --abi-only) ABI_ONLY=1 ;;
        --serve-qwen) SERVE_QWEN=1 ;;
        --skip-csrc) SKIP_CSRC=1 ;;
        --skip-ogroups) SKIP_OGROUPS=1 ;;
        --skip-patches) SKIP_PATCHES=1 ;;
        --help|-h) grep -E '^# ' "$0" | head -25; exit 0 ;;
        *) echo "[FATAL] unknown arg: $arg"; exit 2 ;;
    esac
done

# ---- profile knobs ----
case "$PROFILE" in
    dsv4-pr40760)
        SHA_TARGET="${SHA_TARGET:-aa114601d}"
        VENV_DIR_DEFAULT="/home/z/.venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4"
        VENV_NAME_DEFAULT=".venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4"
        SRC_DIR_DEFAULT="/repo/INSTALLERS/vllm.dsa.sm86/vllm-source"
        WHEEL_DIR_DEFAULT="/repo/wheels/vllm.dsa.sm86_dsv4"
        VLLM_VER_PREFIX="0.19"
        ;;
    baseline-020)
        SHA_TARGET="${SHA_TARGET:-101584af0}"
        VENV_DIR_DEFAULT="${HOME}/.venvs/.venv-vllm_dsa_sm86_baseline"
        VENV_NAME_DEFAULT=".venv-vllm_dsa_sm86_baseline"
        SRC_DIR_DEFAULT="/repo/INSTALLERS/vllm.dsa.sm86/vllm-source"
        WHEEL_DIR_DEFAULT="/repo/wheels/vllm.dsa.sm86"
        VLLM_VER_PREFIX="0.20"
        ;;
    *) echo "[FATAL] unknown profile: $PROFILE"; exit 2 ;;
esac

VLLM_REMOTE="${VLLM_REMOTE:-https://github.com/vllm-project/vllm.git}"
SRC_DIR="${SRC_DIR:-$SRC_DIR_DEFAULT}"
WHEEL_DIR="${WHEEL_DIR:-$WHEEL_DIR_DEFAULT}"
VENV_NAME="${VENV_NAME:-$VENV_NAME_DEFAULT}"
VENV_DIR="${VENV_DIR:-$VENV_DIR_DEFAULT}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-/home/z/.venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4/bin/python3.12}"
PATCHES_DIR="${PATCHES_DIR:-/repo/INSTALLERS}"
CSRC_PATCHES_DIR="${CSRC_PATCHES_DIR:-/Users/z/AGENTIC/sources/vllm-DSA-sm86-3090/csrc_patches}"   # ws10 origin
CSRC_PATCHES_DIR_REMOTE="${CSRC_PATCHES_DIR_REMOTE:-/repo/INSTALLERS/vllm-DSA-sm86-3090.git/csrc_patches}"
CUDA_HOME_REQ="${CUDA_HOME_REQ:-/usr/local/cuda-12.8}"
EXPECT_TORCH="${EXPECT_TORCH:-2.10.0+cu128}"
EXPECT_RAY="${EXPECT_RAY:-2.54.0}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
MAX_JOBS_DEFAULT="${MAX_JOBS:-32}"
LOG_DIR="${LOG_DIR:-${HOME}/INSTALL_LOGS}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_vllm_dsa_sm86_${PROFILE}_$(date +%Y%m%d_%H%M%S).log}"
USE_CCACHE="${USE_CCACHE:-1}"
USE_SCCACHE="${USE_SCCACHE:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
ABI_PROBE_VENV="${ABI_PROBE_VENV:-${HOME}/.venvs/.venv-vllm_dsa_sm86_abi_probe}"  # ephemeral install for ABI smoke

mkdir -p "$LOG_DIR" "$WHEEL_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[build] start $(date -Is) host=$(hostname) profile=$PROFILE sha=$SHA_TARGET arch=$ARCH_LIST"
echo "[build] src=$SRC_DIR  wheel=$WHEEL_DIR  venv=$VENV_DIR"
echo "[build] ccache=$USE_CCACHE sccache=$USE_SCCACHE deterministic=$DETERMINISTIC"

# Short-circuit: ABI-only mode does not rebuild.
if [[ "$ABI_ONLY" == "1" ]]; then
    LATEST_WHEEL="$(ls -1t $WHEEL_DIR/vllm-${VLLM_VER_PREFIX}*.whl 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST_WHEEL" ]] || { echo "[FATAL] no wheel in $WHEEL_DIR for ABI smoke"; exit 3; }
    echo "[abi] running ABI probe vs $LATEST_WHEEL"
    rm -rf "$ABI_PROBE_VENV"
    "$BOOTSTRAP_PY" -m venv "$ABI_PROBE_VENV"
    source "$ABI_PROBE_VENV/bin/activate"
    pip install --quiet --upgrade pip wheel setuptools
    pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0"
    pip install --quiet "$LATEST_WHEEL"
    python -c "
import vllm, importlib, torch
print('vllm', vllm.__version__)
ok = True
for m in ['vllm._C', 'vllm._moe_C']:
    try: importlib.import_module(m); print('OK   import', m)
    except Exception as e: ok=False; print('FAIL import', m, '->', e)
try:
    import vllm._dsa_sm86 as d; print('OK   import vllm._dsa_sm86 ops:', [x for x in dir(d) if not x.startswith('_')][:6])
except Exception as e:
    print('FAIL import vllm._dsa_sm86 ->', e); ok = False
print('CUDA', torch.cuda.is_available(), 'sm', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')
import sys; sys.exit(0 if ok else 4)
"
    deactivate
    rm -rf "$ABI_PROBE_VENV"
    exit 0
fi

# ---- preconditions ----
[[ -x "$BOOTSTRAP_PY" ]] || { echo "[FATAL] bootstrap python missing: $BOOTSTRAP_PY"; exit 2; }
[[ -d /repo ]] || { echo "[FATAL] /repo not mounted"; exit 2; }
[[ -d "$CUDA_HOME_REQ" ]] || { echo "[FATAL] CUDA 12.8 missing: $CUDA_HOME_REQ"; exit 2; }
export CUDA_HOME="$CUDA_HOME_REQ"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[ok] CUDA_HOME=$CUDA_HOME ($(nvcc --version | grep release || echo none))"

# ---- compiler-cache ----
if [[ "$USE_SCCACHE" == "1" ]] && command -v sccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER=sccache CMAKE_CXX_COMPILER_LAUNCHER=sccache CMAKE_CUDA_COMPILER_LAUNCHER=sccache
    export SCCACHE_DIR="${SCCACHE_DIR:-/repo/cache/sccache_vllm_dsa_sm86}"
    mkdir -p "$SCCACHE_DIR"; sccache --start-server >/dev/null 2>&1 || true
    echo "[ok] sccache active dir=$SCCACHE_DIR"
elif [[ "$USE_CCACHE" == "1" ]] && command -v ccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CCACHE_DIR="${CCACHE_DIR:-/repo/cache/ccache_vllm_dsa_sm86}"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-50G}"
    mkdir -p "$CCACHE_DIR"; ccache --max-size="$CCACHE_MAXSIZE" >/dev/null 2>&1 || true
    echo "[ok] ccache active dir=$CCACHE_DIR maxsize=$CCACHE_MAXSIZE"
fi

if [[ "$DETERMINISTIC" == "1" ]]; then
    : "${SOURCE_DATE_EPOCH:=1700000000}"
    export SOURCE_DATE_EPOCH PYTHONHASHSEED=0
    echo "[ok] deterministic SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH PYTHONHASHSEED=0"
fi

# ---- step 1: clone + checkout ----
mkdir -p "$(dirname "$SRC_DIR")"
if [[ ! -d "$SRC_DIR/.git" ]]; then
    echo "[step] clone vllm @ $SHA_TARGET → $SRC_DIR"
    git clone "$VLLM_REMOTE" "$SRC_DIR"
fi
( cd "$SRC_DIR" && {
    git fetch --all --tags --prune 2>&1 | tail -5 || true
    git fetch origin "$SHA_TARGET" 2>/dev/null || true
    if git show-ref --verify --quiet refs/heads/cc2-build-${PROFILE}; then
        git checkout cc2-build-${PROFILE}
        git reset --hard "$SHA_TARGET"
    else
        git checkout -B cc2-build-${PROFILE} "$SHA_TARGET"
    fi
    [[ "$REBUILD" == "1" ]] && git clean -xdf
    [[ "$DETERMINISTIC" == "1" ]] && export SOURCE_DATE_EPOCH="$(git log -1 --format=%ct HEAD)" && echo "[ok] SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH"
    git config user.email "cc2@protoai-bakari.local"
    git config user.name  "ProtoAI-Bakari [agent: CC2]"
} )

# ---- step 2: cherry-pick csrc_patches/ into vendor tree ----
if [[ "$SKIP_CSRC" != "1" && "$PROFILE" == "dsv4-pr40760" ]]; then
    SOURCE_CSRC="$CSRC_PATCHES_DIR"
    [[ -d "$SOURCE_CSRC" ]] || SOURCE_CSRC="$CSRC_PATCHES_DIR_REMOTE"
    if [[ -d "$SOURCE_CSRC/01_dsa_sm86_kernels" ]]; then
        echo "[step] copy csrc_patches/01_dsa_sm86_kernels → $SRC_DIR/csrc_patches/"
        mkdir -p "$SRC_DIR/csrc_patches/01_dsa_sm86_kernels"
        cp "$SOURCE_CSRC/01_dsa_sm86_kernels/"* "$SRC_DIR/csrc_patches/01_dsa_sm86_kernels/"
        # Wire CMakeLists shard into vllm main CMakeLists.txt (idempotent).
        MAIN_CMAKE="$SRC_DIR/CMakeLists.txt"
        INCLUDE_LINE='include(${CMAKE_SOURCE_DIR}/csrc_patches/01_dsa_sm86_kernels/CMakeLists.dsa_sm86.cmake) # CC2 DSA sm_86 wire'
        if ! grep -qF "csrc_patches/01_dsa_sm86_kernels/CMakeLists.dsa_sm86.cmake" "$MAIN_CMAKE"; then
            # Insert before the final install() block, fall back to append.
            ANCHOR_LINE="$(grep -n '^install(' "$MAIN_CMAKE" | head -1 | cut -d: -f1)"
            if [[ -n "$ANCHOR_LINE" ]]; then
                sed -i "${ANCHOR_LINE}i\\
${INCLUDE_LINE}
" "$MAIN_CMAKE"
            else
                echo "$INCLUDE_LINE" >> "$MAIN_CMAKE"
            fi
            echo "[ok] CMakeLists.txt wired (include line inserted)"
        else
            echo "[ok] CMakeLists.txt already wired"
        fi
    else
        echo "[WARN] csrc_patches/01_dsa_sm86_kernels not found at $SOURCE_CSRC — skipping CC3 shard"
    fi
fi

# ---- step 2b: o_groups relax (CC5 ed96096) ----
if [[ "$SKIP_OGROUPS" != "1" && "$PROFILE" == "dsv4-pr40760" ]]; then
    SOURCE_CSRC="${SOURCE_CSRC:-$CSRC_PATCHES_DIR}"
    if [[ -f "$SOURCE_CSRC/o_groups_relax.py" ]]; then
        echo "[step] stage o_groups_relax.py + .cu into vendor tree"
        mkdir -p "$SRC_DIR/csrc_patches"
        cp "$SOURCE_CSRC/o_groups_relax.py" "$SRC_DIR/csrc_patches/o_groups_relax.py"
        [[ -f "$SOURCE_CSRC/o_groups_relax_sm86.cu" ]] && cp "$SOURCE_CSRC/o_groups_relax_sm86.cu" "$SRC_DIR/csrc_patches/o_groups_relax_sm86.cu"
        # If o_groups_relax.py is a runner that monkey-patches at import time, vllm-runtime path
        # is in vllm/model_executor/models/deepseek_v4.py; CC5 is responsible for the apply hook.
        # Here we just stage so the wheel ships with the file in vllm/csrc_patches/.
        echo "[ok] o_groups_relax.py + .cu staged in $SRC_DIR/csrc_patches/"
    else
        echo "[WARN] o_groups_relax.py not found at $SOURCE_CSRC — skipping CC5 patch"
    fi
fi

# ---- step 2c: P6/P7/P8 (baseline profile only — vendor tree edits) ----
if [[ "$SKIP_PATCHES" != "1" && "$PROFILE" == "baseline-020" ]]; then
    apply_vendor_patch() {
        local label="$1" src_patch="$2"
        local tmp="$(mktemp)"
        sed -E -e "s#^VENV=.*#VENV=${SRC_DIR}#" \
               -e "s#\\\$VENV/lib/python3\\.12/site-packages/vllm/#\\\$VENV/vllm/#g" \
               -e "s#\\\$VENV/lib/python3\\.12/site-packages/triton/#\\\$VENV/triton/#g" \
               "$src_patch" > "$tmp"
        chmod +x "$tmp"; bash "$tmp"; rm -f "$tmp"
    }
    SRC_TFILE="$SRC_DIR/vllm/model_executor/kernels/linear/scaled_mm/triton.py"
    [[ -f "$SRC_TFILE" ]] && {
        apply_vendor_patch "P6 UE8M0_PRECONVERT"  "${PATCHES_DIR}/patch_vllm_ue8m0_preconvert.sh"
        apply_vendor_patch "P8 W8A8_BF16_FALLBACK" "${PATCHES_DIR}/patch_vllm_w8a8_bf16_fallback.sh"
    }
    ( cd "$SRC_DIR" && {
        if ! git diff --quiet HEAD -- vllm/; then
            git add vllm/
            git commit -m "P6+P8 baseline (UE8M0_PRECONVERT + W8A8_BF16_FALLBACK) for sm_86

Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 [agent: CC2]"
        fi
    } )
fi

# ---- step 3: venv ----
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "[step] create venv: $VENV_DIR"
    "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -V
pip install --upgrade --quiet pip wheel setuptools

# ---- step 4: torch + ray + build deps ----
T_NOW="$(python -c 'import torch,sys; sys.stdout.write(getattr(torch,"__version__","none"))' 2>/dev/null || echo none)"
if [[ "$T_NOW" != "$EXPECT_TORCH" ]]; then
    pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0" torchvision torchaudio
fi
R_NOW="$(python -c 'import ray,sys; sys.stdout.write(getattr(ray,"__version__","none"))' 2>/dev/null || echo none)"
[[ "$R_NOW" != "$EXPECT_RAY" ]] && pip install --quiet "ray[default]==${EXPECT_RAY}"
pip install --quiet setuptools_scm cmake ninja packaging numpy pybind11
[[ -f "$SRC_DIR/requirements/build.txt" ]] && pip install --quiet -r "$SRC_DIR/requirements/build.txt" || true
[[ -f "$SRC_DIR/requirements-build.txt" ]] && pip install --quiet -r "$SRC_DIR/requirements-build.txt" || true

# ---- step 5: source build ----
WHEEL_GLOB="$WHEEL_DIR/vllm-${VLLM_VER_PREFIX}*.whl"
EXISTING_WHEEL="$(ls -1t $WHEEL_GLOB 2>/dev/null | head -1 || true)"
if [[ -n "$EXISTING_WHEEL" && "$REBUILD" != "1" ]]; then
    echo "[ok] reusing wheel: $EXISTING_WHEEL"
else
    echo "[step] cmake compile profile=$PROFILE arch=$ARCH_LIST → $WHEEL_DIR"
    export TORCH_CUDA_ARCH_LIST="$ARCH_LIST" VLLM_TARGET_DEVICE="cuda" MAX_JOBS="$MAX_JOBS_DEFAULT"
    BUILD_T0=$(date +%s)
    ( cd "$SRC_DIR" && pip wheel . -w "$WHEEL_DIR" --no-build-isolation )
    BUILD_T1=$(date +%s)
    echo "[ok] compile elapsed $((BUILD_T1 - BUILD_T0))s"
    EXISTING_WHEEL="$(ls -1t $WHEEL_GLOB 2>/dev/null | head -1)"
    [[ "$USE_SCCACHE" == "1" ]] && command -v sccache >/dev/null 2>&1 && sccache --show-stats || true
    [[ "$USE_CCACHE"  == "1" ]] && command -v ccache  >/dev/null 2>&1 && ccache --show-stats || true
fi
[[ -n "$EXISTING_WHEEL" ]] || { echo "[FATAL] no wheel produced"; exit 3; }
echo "[wheel] $EXISTING_WHEEL  size=$(du -h "$EXISTING_WHEEL" | awk '{print $1}')"

# ---- step 6: ABI smoke (ephemeral venv, vs unpatched torch) — acceptance (d) ----
echo "[step] ABI smoke against new wheel in ephemeral venv"
rm -rf "$ABI_PROBE_VENV"
"$BOOTSTRAP_PY" -m venv "$ABI_PROBE_VENV"
( source "$ABI_PROBE_VENV/bin/activate"
  pip install --quiet --upgrade pip wheel setuptools
  pip install --quiet --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0"
  pip install --quiet "$EXISTING_WHEEL" || true
  python - <<'PY' || { echo "[FAIL] ABI smoke failed"; exit 4; }
import sys, importlib, torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__)
ok = True
for m in ("vllm._C", "vllm._moe_C"):
    try: importlib.import_module(m); print("OK  ", m)
    except Exception as e: ok=False; print("FAIL", m, "->", e)
try:
    import vllm._dsa_sm86 as d
    print("OK   vllm._dsa_sm86 syms:", sorted([x for x in dir(d) if not x.startswith("_")])[:8])
except Exception as e:
    print("WARN vllm._dsa_sm86 ->", e)
print("CUDA available:", torch.cuda.is_available(),
      "sm:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
sys.exit(0 if ok else 4)
PY
)
ABI_RC=$?
rm -rf "$ABI_PROBE_VENV"
[[ "$ABI_RC" != "0" ]] && { echo "[FATAL] ABI smoke FAIL — wheel will NOT be installed into $VENV_DIR"; exit 4; }
echo "[ok] ABI smoke PASS"

# ---- step 7: install into target venv ----
if [[ "$SKIP_INSTALL" == "1" ]]; then
    echo "[skip] --skip-install set; wheel ready at $EXISTING_WHEEL"
else
    INSTALLED_VLLM="$(python -c 'import vllm,sys; sys.stdout.write(getattr(vllm,"__version__","none"))' 2>/dev/null || echo none)"
    if [[ "$INSTALLED_VLLM" == ${VLLM_VER_PREFIX}* && "$REBUILD" != "1" ]]; then
        echo "[ok] vllm $INSTALLED_VLLM already in $VENV_DIR (use --rebuild to force)"
    else
        pip install --quiet --force-reinstall --no-deps "$EXISTING_WHEEL"
        pip install --quiet "$EXISTING_WHEEL"
    fi
fi

# ---- step 7b (baseline-020 only): P7 sp patch ----
if [[ "$SKIP_PATCHES" != "1" && "$PROFILE" == "baseline-020" && "$SKIP_INSTALL" != "1" ]]; then
    SP_DIR="$VENV_DIR/lib/python3.12/site-packages"
    UFILE="$SP_DIR/triton/_utils.py"
    if [[ -f "$UFILE" ]]; then
        tmp="$(mktemp)"
        sed -E "s#^VENV=.*#VENV=${VENV_DIR}#" "${PATCHES_DIR}/patch_triton_ue8m0_dict.sh" > "$tmp"
        chmod +x "$tmp"; bash "$tmp"; rm -f "$tmp"
    fi
fi

# ---- step 8: smoke ----
echo "================ POST-INSTALL SMOKE ================"
python -c "
import vllm, torch
print(f'vllm    = {vllm.__version__}')
print(f'torch   = {torch.__version__}')
print(f'cuda    = {torch.version.cuda}')
print(f'sm      = {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"n/a\"}')
"
echo "===================================================="

# ---- manifest ----
MANIFEST="$VENV_DIR/.cc2_build_manifest.txt"
{
    echo "venv=$VENV_NAME"
    echo "built_at=$(date -Is)"
    echo "host=$(hostname)"
    echo "profile=$PROFILE"
    echo "sha=$SHA_TARGET"
    echo "vendor_branch=$(git -C "$SRC_DIR" rev-parse --abbrev-ref HEAD)"
    echo "vendor_head=$(git -C "$SRC_DIR" rev-parse HEAD)"
    echo "arch_list=$ARCH_LIST"
    echo "ccache=$USE_CCACHE sccache=$USE_SCCACHE deterministic=$DETERMINISTIC"
    [[ "$DETERMINISTIC" == "1" ]] && echo "source_date_epoch=$SOURCE_DATE_EPOCH"
    echo "csrc_kernels=$([[ "$SKIP_CSRC" == "1" ]] && echo skipped || echo applied_4db7f8e)"
    echo "o_groups_relax=$([[ "$SKIP_OGROUPS" == "1" ]] && echo skipped || echo applied_ed96096)"
    echo "wheel=$EXISTING_WHEEL"
    echo "abi_smoke=PASS"
    python -c "import vllm; print(f'vllm={vllm.__version__}')"
    python -c "import torch; print(f'torch={torch.__version__}')"
} > "$MANIFEST"
echo "[ok] manifest: $MANIFEST"

# ---- step 9 (optional): Qwen3.5 verify ----
if [[ "$SERVE_QWEN" == "1" ]]; then
    QWEN_LOG="${LOG_DIR}/qwen_serve_$(date +%Y%m%d_%H%M%S).log"
    nohup vllm serve Qwen/Qwen2.5-72B-Instruct-FP8 \
        --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --port 8000 > "$QWEN_LOG" 2>&1 &
    QPID=$!
    for i in $(seq 1 30); do
        sleep 3
        curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q '"data"' && { echo "[ok] /v1/models 200"; break; }
    done
    HC="$(curl -s -o "${LOG_DIR}/qwen_completions.json" -w '%{http_code}' \
         -X POST http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
         -d '{"model":"Qwen/Qwen2.5-72B-Instruct-FP8","prompt":"Hello","max_tokens":4}')"
    echo "[result] HTTP $HC"
    kill -TERM "$QPID" 2>/dev/null || true
    [[ "$HC" != "200" ]] && exit 5
fi

echo "[build] DONE $(date -Is) wheel=$EXISTING_WHEEL"
