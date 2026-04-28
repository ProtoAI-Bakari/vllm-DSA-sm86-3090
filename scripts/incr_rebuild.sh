#!/usr/bin/env bash
# incr_rebuild.sh — CC2 Story 7: per-kernel incremental rebuild.
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC2]
#
# Use case: CC3/CC4/CC5 iterates on a single .cu / .cpp / .py file. Full vllm pip wheel
# rebuild is 30-60 min cold. With ccache it's 3-10 min. With direct ninja --target on
# a single shared lib it can be ~30s-3min.
#
# This script:
#   1. Copies provided source file(s) into the vendor tree (vllm-source/csrc_patches/...).
#   2. Runs `cmake --build` against ONLY the target shared lib that depends on those files.
#   3. Locates the produced .so and reinstalls it into the target venv (no pip wheel).
#   4. Runs ABI smoke (vllm._dsa_sm86 + vllm._C imports).
#
# Use:
#   bash scripts/incr_rebuild.sh <src1> [<src2> ...]
#   bash scripts/incr_rebuild.sh --target _dsa_sm86 csrc_patches/01_dsa_sm86_kernels/sparse_attn_indexer_sm86.cu
#   bash scripts/incr_rebuild.sh --target _C csrc/some_kernel.cu  (for CC3 ports inside vendor csrc)
#   bash scripts/incr_rebuild.sh --venv DIR
#   bash scripts/incr_rebuild.sh --no-install  (build only, no .so swap)
#
# Walls: median ~60-180s warm ccache; cold first-iter ~5min for _dsa_sm86 (4 sources).

set -uo pipefail

SRC_DIR="${SRC_DIR:-/repo/INSTALLERS/vllm.dsa.sm86/vllm-source}"
TARGET="${TARGET:-_dsa_sm86}"
VENV_DIR="${VENV_DIR:-/home/z/.venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
CUDA_HOME_REQ="${CUDA_HOME_REQ:-/usr/local/cuda-12.8}"
LOG_DIR="${LOG_DIR:-${HOME}/INSTALL_LOGS}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/incr_rebuild_${TARGET}_$(date +%Y%m%d_%H%M%S).log}"
USE_CCACHE="${USE_CCACHE:-1}"
DO_INSTALL=1
DO_SMOKE=1
SOURCES=()
for arg in "$@"; do
    case "$arg" in
        --target) shift; TARGET="$1" ;;
        --target=*) TARGET="${arg#--target=}" ;;
        --venv) shift; VENV_DIR="$1" ;;
        --venv=*) VENV_DIR="${arg#--venv=}" ;;
        --no-install) DO_INSTALL=0 ;;
        --no-smoke) DO_SMOKE=0 ;;
        --help|-h) grep -E '^# ' "$0" | head -25; exit 0 ;;
        *) SOURCES+=("$arg") ;;
    esac
done

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[incr] start $(date -Is) target=$TARGET venv=$VENV_DIR sources=${#SOURCES[@]}"

[[ -d "$SRC_DIR/.git" ]] || { echo "[FATAL] vendor tree missing: $SRC_DIR (run build_vllm_dsa_sm86.sh first)"; exit 2; }
[[ -d "$CUDA_HOME_REQ" ]] || { echo "[FATAL] CUDA 12.8 missing: $CUDA_HOME_REQ"; exit 2; }
export CUDA_HOME="$CUDA_HOME_REQ"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

if [[ "$USE_CCACHE" == "1" ]] && command -v ccache >/dev/null 2>&1; then
    export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CCACHE_DIR="${CCACHE_DIR:-/repo/cache/ccache_vllm_dsa_sm86}"
    mkdir -p "$CCACHE_DIR"
fi

# ---- step 1: stage sources into vendor tree ----
for s in "${SOURCES[@]}"; do
    [[ -f "$s" ]] || { echo "[FATAL] source not found: $s"; exit 3; }
    base="$(basename "$s")"
    case "$base" in
        *.cu|*.cuh|*.cpp|*.h|*.hpp)
            dest="$SRC_DIR/csrc_patches/01_dsa_sm86_kernels/$base"
            ;;
        *.cmake)
            dest="$SRC_DIR/csrc_patches/01_dsa_sm86_kernels/$base"
            ;;
        *.py)
            # Python source: stage to csrc_patches/, do not require ninja rebuild for pure-py.
            dest="$SRC_DIR/csrc_patches/$base"
            ;;
        *) echo "[WARN] unrecognized ext for $s — staging to csrc_patches/" ; dest="$SRC_DIR/csrc_patches/$base" ;;
    esac
    mkdir -p "$(dirname "$dest")"
    cp "$s" "$dest"
    touch "$dest"
    echo "[stage] $s → $dest"
done

# Pure-py iteration: no ninja rebuild needed.
PURE_PY=1
for s in "${SOURCES[@]}"; do
    case "$(basename "$s")" in
        *.cu|*.cuh|*.cpp|*.h|*.hpp|*.cmake) PURE_PY=0 ;;
    esac
done

if [[ "$PURE_PY" == "1" && "$DO_INSTALL" == "1" ]]; then
    echo "[step] pure-py iter — copy .py into installed site-packages"
    SP_DIR="$VENV_DIR/lib/python3.12/site-packages/vllm/csrc_patches"
    mkdir -p "$SP_DIR"
    for s in "${SOURCES[@]}"; do
        cp "$s" "$SP_DIR/"
        echo "[install] $s → $SP_DIR/$(basename "$s")"
    done
    exit 0
fi

# ---- step 2: locate or create build dir ----
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build_incr}"
if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
    echo "[step] cmake configure (one-time) → $BUILD_DIR"
    source "$VENV_DIR/bin/activate"
    cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="86" \
        -DTORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
        -DVLLM_TARGET_DEVICE=cuda \
        -DPython_EXECUTABLE="$VENV_DIR/bin/python" \
        -DCMAKE_PREFIX_PATH="$($VENV_DIR/bin/python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"
fi

# ---- step 3: cmake --build target ----
echo "[step] cmake --build $BUILD_DIR --target $TARGET"
T0=$(date +%s)
cmake --build "$BUILD_DIR" --target "$TARGET" -j "${MAX_JOBS:-32}"
RC=$?
T1=$(date +%s)
echo "[ok] build elapsed $((T1 - T0))s rc=$RC"
[[ "$RC" != "0" ]] && exit "$RC"

# ---- step 4: locate produced .so ----
SO_PATH="$(find "$BUILD_DIR" -maxdepth 4 -name "${TARGET}*.so" -type f 2>/dev/null | head -1)"
if [[ -z "$SO_PATH" ]]; then
    SO_PATH="$(find "$BUILD_DIR" -maxdepth 4 -name "*${TARGET}*.so" -type f 2>/dev/null | head -1)"
fi
[[ -z "$SO_PATH" ]] && { echo "[FATAL] no .so produced for target $TARGET in $BUILD_DIR"; exit 4; }
echo "[so] $SO_PATH ($(du -h "$SO_PATH" | awk '{print $1}'))"

# ---- step 5: install .so into venv ----
if [[ "$DO_INSTALL" == "1" ]]; then
    SP_VLLM="$VENV_DIR/lib/python3.12/site-packages/vllm"
    [[ -d "$SP_VLLM" ]] || { echo "[FATAL] target venv missing vllm: $SP_VLLM"; exit 5; }
    SO_BASENAME="$(basename "$SO_PATH")"
    cp -f "$SO_PATH" "$SP_VLLM/$SO_BASENAME"
    echo "[install] $SO_PATH → $SP_VLLM/$SO_BASENAME"
fi

# ---- step 6: smoke ----
if [[ "$DO_SMOKE" == "1" ]]; then
    "$VENV_DIR/bin/python" - <<PY
import importlib, sys
ok = True
for m in ("vllm", "vllm._C", "vllm._moe_C", "vllm._dsa_sm86"):
    try: importlib.import_module(m); print("OK  ", m)
    except Exception as e: ok=False; print("FAIL", m, "->", e)
sys.exit(0 if ok else 6)
PY
    [[ "$?" != "0" ]] && exit 6
fi

echo "[incr] DONE $(date -Is) target=$TARGET wall=$((T1 - T0))s"
