#!/usr/bin/env bash
# build_vllm_dsa_sm86.sh — CC2 baseline source rebuild of vLLM 0.20.0 for sm_86 (Ampere RTX 3090).
#
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC2]
#
# What this does (vanilla baseline — NO new DSA kernel ports yet):
#   1. Clone vllm-project/vllm at SHA 101584af0 (matches deployed .venv-vllm_0_20_0_fresh) into
#      /repo/INSTALLERS/vllm.dsa.sm86/vllm-source/.
#   2. Author/refresh sibling venv at /home/z/.venvs/.venv-vllm_dsa_sm86_baseline/ to avoid
#      clobbering the canonical .venv-vllm_0_20_0_fresh. Bootstrap python = PR40760's working
#      python3.12.
#   3. Pip-install torch 2.10.0+cu128 + ray 2.54.0 + build deps.
#   4. cmake/ninja source build via `pip wheel . -w /repo/wheels/vllm.dsa.sm86 --no-build-isolation`
#      with CUDA_HOME=/usr/local/cuda-12.8 (system nvcc 12.0 lacks CU_MEM_HANDLE_TYPE_FABRIC),
#      TORCH_CUDA_ARCH_LIST="8.6", VLLM_TARGET_DEVICE=cuda, MAX_JOBS=32.
#   5. Install built wheel into baseline venv.
#   6. Apply P6 (UE8M0_PRECONVERT) + P7 (UE8M0_SCALE_SHIM) + P8 (W8A8_BF16_FALLBACK) to
#      installed site-packages, with hardcoded old-venv paths rewritten via CC2_VENV_OVERRIDE
#      (the patches respect $VENV env var if set; otherwise we apply them to a venv-path-rewritten
#      copy under a temp dir).
#   7. Smoke test: import vllm; print version + cuda capability.
#   8. Optional --serve-qwen flag: launch `vllm serve Qwen/Qwen2.5-72B-Instruct-FP8` and curl
#      :8000/v1/completions to verify HTTP 200 (only run by CC9 broker on cuda1 with --serve-qwen).
#
# This script is idempotent: re-run safely. Each step short-circuits on already-done state.
#
# Wall: clone+venv ~5 min; pip wheel cmake/ninja build ~30-60 min on cuda host CPU; smoke ~1 min.
#
# Failure modes (known from prior P6/P7/P8 + PR40760 history):
#   - nvcc 12.0 vs 12.8: CU_MEM_HANDLE_TYPE_FABRIC missing → activation_kernels.cu won't compile.
#     Fixed by sourcing CUDA 12.8 first.
#   - flashinfer-cubin pre-built bundle: vLLM 0.20.0 ships with flashinfer wheel deps, not built
#     from source here. Stays as wheel.
#   - quack-kernels, deep-gemm: pip-installed; we don't recompile these.

set -euo pipefail

# ---- knobs ----
SHA_TARGET="${SHA_TARGET:-101584af0}"          # vLLM 0.20.0 commit_id from deployed venv
VLLM_REMOTE="${VLLM_REMOTE:-https://github.com/vllm-project/vllm.git}"
SRC_DIR="${SRC_DIR:-/repo/INSTALLERS/vllm.dsa.sm86/vllm-source}"
WHEEL_DIR="${WHEEL_DIR:-/repo/wheels/vllm.dsa.sm86}"
VENV_NAME="${VENV_NAME:-.venv-vllm_dsa_sm86_baseline}"
VENV_DIR="${VENV_DIR:-${HOME}/.venvs/${VENV_NAME}}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-/home/z/.venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4/bin/python3.12}"
PATCHES_DIR="${PATCHES_DIR:-/repo/INSTALLERS}"
CUDA_HOME_REQ="${CUDA_HOME_REQ:-/usr/local/cuda-12.8}"
EXPECT_TORCH="${EXPECT_TORCH:-2.10.0+cu128}"
EXPECT_RAY="${EXPECT_RAY:-2.54.0}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
MAX_JOBS_DEFAULT="${MAX_JOBS:-32}"
LOG_DIR="${LOG_DIR:-${HOME}/INSTALL_LOGS}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/build_vllm_dsa_sm86_$(date +%Y%m%d_%H%M%S).log}"

SERVE_QWEN=0
REBUILD=0
SKIP_PATCHES=0
for arg in "$@"; do
    case "$arg" in
        --serve-qwen) SERVE_QWEN=1 ;;
        --rebuild)    REBUILD=1 ;;
        --skip-patches) SKIP_PATCHES=1 ;;
        --help|-h)
            grep -E '^# ' "$0" | head -50
            exit 0
            ;;
        *) echo "[FATAL] unknown arg: $arg"; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR" "$WHEEL_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[build_vllm_dsa_sm86] start $(date -Is) host=$(hostname) sha=$SHA_TARGET arch=$ARCH_LIST"
echo "[build_vllm_dsa_sm86] src=$SRC_DIR  wheel=$WHEEL_DIR  venv=$VENV_DIR"

# ---- preconditions ----
[[ -x "$BOOTSTRAP_PY" ]] || { echo "[FATAL] bootstrap python missing: $BOOTSTRAP_PY"; exit 2; }
[[ -d /repo ]] || { echo "[FATAL] /repo not mounted"; exit 2; }
if [[ ! -d "$CUDA_HOME_REQ" ]]; then
    echo "[FATAL] CUDA 12.8 toolchain missing at $CUDA_HOME_REQ. System nvcc 12.0 lacks CU_MEM_HANDLE_TYPE_FABRIC and breaks vllm 0.20.0 build."
    exit 2
fi

export CUDA_HOME="$CUDA_HOME_REQ"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "[ok] CUDA_HOME=$CUDA_HOME ($(nvcc --version | grep release || echo 'nvcc missing'))"

# ---- step 1: clone vllm @ SHA ----
mkdir -p "$(dirname "$SRC_DIR")"
if [[ ! -d "$SRC_DIR/.git" ]]; then
    echo "[step] cloning vllm @ $SHA_TARGET into $SRC_DIR"
    git clone "$VLLM_REMOTE" "$SRC_DIR"
fi
( cd "$SRC_DIR" && {
    git fetch --all --tags --prune
    if ! git rev-parse --verify "$SHA_TARGET" >/dev/null 2>&1; then
        # Try fetching the specific commit by partial SHA (may fail on shallow remote)
        git fetch origin "$SHA_TARGET" 2>/dev/null || true
    fi
    CURRENT_SHA="$(git rev-parse HEAD)"
    if [[ "$CURRENT_SHA" != "${SHA_TARGET}"* ]]; then
        echo "[step] checking out $SHA_TARGET"
        git checkout -B cc2-build "$SHA_TARGET"
    else
        echo "[ok] already at $SHA_TARGET"
    fi
    if [[ "$REBUILD" == "1" ]]; then
        echo "[step] --rebuild → git clean -xdf"
        git clean -xdf
    fi
} )

# ---- step 2: venv ----
if [[ -x "$VENV_DIR/bin/python" ]]; then
    echo "[ok] venv already exists at $VENV_DIR"
else
    echo "[step] creating fresh venv with $BOOTSTRAP_PY"
    "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
python -V
pip install --upgrade --quiet pip wheel setuptools

# ---- step 3: torch + ray pin ----
T_NOW="$(python -c 'import torch,sys; sys.stdout.write(getattr(torch,"__version__","none"))' 2>/dev/null || echo none)"
if [[ "$T_NOW" != "$EXPECT_TORCH" ]]; then
    echo "[step] installing torch $EXPECT_TORCH"
    pip install --quiet --index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.10.0" torchvision torchaudio
else
    echo "[ok] torch $EXPECT_TORCH already installed"
fi
R_NOW="$(python -c 'import ray,sys; sys.stdout.write(getattr(ray,"__version__","none"))' 2>/dev/null || echo none)"
if [[ "$R_NOW" != "$EXPECT_RAY" ]]; then
    echo "[step] installing ray==$EXPECT_RAY"
    pip install --quiet "ray[default]==${EXPECT_RAY}"
else
    echo "[ok] ray $EXPECT_RAY already installed"
fi

# ---- step 4: build deps ----
pip install --quiet setuptools_scm cmake ninja packaging numpy pybind11
if [[ -f "$SRC_DIR/requirements/build.txt" ]]; then
    pip install --quiet -r "$SRC_DIR/requirements/build.txt"
elif [[ -f "$SRC_DIR/requirements-build.txt" ]]; then
    pip install --quiet -r "$SRC_DIR/requirements-build.txt"
fi

# ---- step 5: source build (cmake/ninja via pip wheel) ----
WHEEL_GLOB="$WHEEL_DIR/vllm-0.20.0*.whl"
EXISTING_WHEEL="$(ls -1t $WHEEL_GLOB 2>/dev/null | head -1 || true)"
if [[ -n "$EXISTING_WHEEL" && "$REBUILD" != "1" ]]; then
    echo "[ok] reusing pre-built wheel: $EXISTING_WHEEL"
else
    echo "[step] cmake/ninja source build → $WHEEL_DIR"
    export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
    export VLLM_TARGET_DEVICE="cuda"
    export MAX_JOBS="$MAX_JOBS_DEFAULT"
    ( cd "$SRC_DIR" && pip wheel . -w "$WHEEL_DIR" --no-build-isolation )
    EXISTING_WHEEL="$(ls -1t $WHEEL_GLOB 2>/dev/null | head -1)"
fi
[[ -n "$EXISTING_WHEEL" ]] || { echo "[FATAL] no wheel produced in $WHEEL_DIR"; exit 3; }

# ---- step 6: install wheel ----
INSTALLED_VLLM="$(python -c 'import vllm,sys; sys.stdout.write(getattr(vllm,"__version__","none"))' 2>/dev/null || echo none)"
if [[ "$INSTALLED_VLLM" == 0.20.* && "$REBUILD" != "1" ]]; then
    echo "[ok] vllm $INSTALLED_VLLM already installed in $VENV_DIR"
else
    echo "[step] pip install $EXISTING_WHEEL"
    pip install --quiet --force-reinstall --no-deps "$EXISTING_WHEEL"
    pip install --quiet "$EXISTING_WHEEL"
fi

# ---- step 7: apply P6/P7/P8 to installed site-packages with venv-path rewrite ----
SP_DIR="$VENV_DIR/lib/python3.12/site-packages"
TFILE="$SP_DIR/vllm/model_executor/kernels/linear/scaled_mm/triton.py"
UFILE="$SP_DIR/triton/_utils.py"

apply_p_patch() {
    local label="$1" src_patch="$2" target="$3"
    local tmp_patch
    tmp_patch="$(mktemp)"
    # Rewrite hardcoded VENV path inside the patch script to our baseline venv.
    sed -E "s#^VENV=.*#VENV=${VENV_DIR}#" "$src_patch" > "$tmp_patch"
    chmod +x "$tmp_patch"
    echo "[step] applying $label ($src_patch) → ${target}"
    bash "$tmp_patch"
    rm -f "$tmp_patch"
}

if [[ "$SKIP_PATCHES" != "1" ]]; then
    [[ -f "$TFILE" ]] || { echo "[FATAL] $TFILE missing post-install"; exit 4; }
    [[ -f "$UFILE" ]] || { echo "[FATAL] $UFILE missing post-install"; exit 4; }
    apply_p_patch "P6 UE8M0_PRECONVERT"  "${PATCHES_DIR}/patch_vllm_ue8m0_preconvert.sh"  "$TFILE"
    apply_p_patch "P7 UE8M0_SCALE_SHIM"   "${PATCHES_DIR}/patch_triton_ue8m0_dict.sh"      "$UFILE"
    apply_p_patch "P8 W8A8_BF16_FALLBACK" "${PATCHES_DIR}/patch_vllm_w8a8_bf16_fallback.sh" "$TFILE"
else
    echo "[skip] --skip-patches set — P6/P7/P8 NOT applied"
fi

# ---- step 8: smoke ----
echo
echo "================ POST-BUILD SMOKE ================"
python -c "
import vllm, torch, triton
print(f'vllm    = {vllm.__version__}')
print(f'torch   = {torch.__version__}')
print(f'triton  = {triton.__version__}')
print(f'cuda    = {torch.version.cuda}')
print(f'sm      = {torch.cuda.get_device_capability(0)}')
print(f'gpu     = {torch.cuda.get_device_name(0)}')
"
echo "==================================================="

# ---- record manifest ----
MANIFEST="$VENV_DIR/.cc2_build_manifest.txt"
{
    echo "venv=$VENV_NAME"
    echo "built_at=$(date -Is)"
    echo "host=$(hostname)"
    echo "sha=$SHA_TARGET"
    echo "arch_list=$ARCH_LIST"
    echo "patches_applied=$([[ "$SKIP_PATCHES" == "1" ]] && echo 'none' || echo 'P6 P7 P8')"
    python -c "import vllm; print(f'vllm={vllm.__version__}')"
    python -c "import torch; print(f'torch={torch.__version__}')"
    python -c "import triton; print(f'triton={triton.__version__}')"
} > "$MANIFEST"
echo "[ok] manifest: $MANIFEST"

# ---- step 9 (optional): serve Qwen verification ----
if [[ "$SERVE_QWEN" == "1" ]]; then
    echo "[step] launching vllm serve Qwen/Qwen2.5-72B-Instruct-FP8 (background)"
    QWEN_LOG="${LOG_DIR}/qwen_serve_$(date +%Y%m%d_%H%M%S).log"
    nohup vllm serve Qwen/Qwen2.5-72B-Instruct-FP8 \
        --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --port 8000 > "$QWEN_LOG" 2>&1 &
    QWEN_PID=$!
    echo "[ok] vllm serve PID=$QWEN_PID log=$QWEN_LOG"
    echo "[step] waiting 90s for vllm serve to come up"
    for i in $(seq 1 30); do
        sleep 3
        if curl -s -m 2 http://127.0.0.1:8000/v1/models | grep -q '"data"'; then
            echo "[ok] /v1/models 200 after ${i}*3s"
            break
        fi
        if [[ "$i" == "30" ]]; then
            echo "[WARN] /v1/models did not become ready in 90s; continuing"
        fi
    done
    echo "[step] curl /v1/completions"
    HTTP_CODE="$(curl -s -o "${LOG_DIR}/qwen_completions.json" -w '%{http_code}' \
        -X POST http://127.0.0.1:8000/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"Qwen/Qwen2.5-72B-Instruct-FP8","prompt":"Hello","max_tokens":4}')"
    echo "[result] HTTP $HTTP_CODE  (body in ${LOG_DIR}/qwen_completions.json)"
    if [[ "$HTTP_CODE" != "200" ]]; then
        echo "[FAIL] /v1/completions returned $HTTP_CODE"
        kill -TERM "$QWEN_PID" 2>/dev/null || true
        exit 5
    fi
    kill -TERM "$QWEN_PID" 2>/dev/null || true
fi

echo "[build_vllm_dsa_sm86] DONE $(date -Is)"
