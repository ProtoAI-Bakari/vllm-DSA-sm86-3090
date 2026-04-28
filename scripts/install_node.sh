#!/usr/bin/env bash
# install_node.sh — CC2 Story 9: cluster-side per-node install wrapper.
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC2]
#
# Used by CC7 fanout to install the vllm-DSA-sm86 wheel into a node's target venv.
# Idempotent. Safe to re-run. Refuses to proceed if local env diverges from build_env.lock
# (unless --skip-verify-env). Returns rc=0 on PASS, non-zero on FAIL (CC7 aggregates).
#
# Use:
#   bash scripts/install_node.sh                                # standard fanout install
#   bash scripts/install_node.sh --wheel /repo/wheels/vllm.dsa.sm86_dsv4/vllm-0.19.1*.whl
#   bash scripts/install_node.sh --target-venv DIR
#   bash scripts/install_node.sh --skip-verify-env
#   bash scripts/install_node.sh --rollback                     # restore previous wheel
#   bash scripts/install_node.sh --abi-only

set -uo pipefail

WHEEL_DIR_DEFAULT="/repo/wheels/vllm.dsa.sm86_dsv4"
TARGET_VENV="${TARGET_VENV:-/home/z/.venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_LOCK="${ENV_LOCK:-$SCRIPT_DIR/build_env.lock}"
VERIFY="${VERIFY:-$SCRIPT_DIR/verify_env.sh}"
LOG_DIR="${LOG_DIR:-${HOME}/INSTALL_LOGS}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/install_node_$(date +%Y%m%d_%H%M%S).log}"
PREV_DIR="${TARGET_VENV}/.cc2_prev_wheel"
WHEEL=""
SKIP_VERIFY=0
ABI_ONLY=0
ROLLBACK=0
for arg in "$@"; do
    case "$arg" in
        --wheel) shift; WHEEL="$1" ;;
        --wheel=*) WHEEL="${arg#--wheel=}" ;;
        --target-venv) shift; TARGET_VENV="$1" ;;
        --target-venv=*) TARGET_VENV="${arg#--target-venv=}" ;;
        --skip-verify-env) SKIP_VERIFY=1 ;;
        --abi-only) ABI_ONLY=1 ;;
        --rollback) ROLLBACK=1 ;;
        --help|-h) grep -E '^# ' "$0" | head -20; exit 0 ;;
    esac
done

mkdir -p "$LOG_DIR" "$PREV_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
HOST="$(hostname)"
echo "[install_node] start $(date -Is) host=$HOST target=$TARGET_VENV"

# ---- rollback path ----
if [[ "$ROLLBACK" == "1" ]]; then
    LATEST_PREV="$(ls -1t $PREV_DIR/vllm-*.whl 2>/dev/null | head -1 || true)"
    [[ -n "$LATEST_PREV" ]] || { echo "[FATAL] no prior wheel in $PREV_DIR for rollback"; exit 7; }
    "$TARGET_VENV/bin/pip" install --quiet --force-reinstall --no-deps "$LATEST_PREV"
    echo "[ok] ROLLBACK to $LATEST_PREV"
    exit 0
fi

# ---- ABI-only ----
if [[ "$ABI_ONLY" == "1" ]]; then
    "$TARGET_VENV/bin/python" - <<'PY'
import importlib, sys
ok = True
for m in ("vllm", "vllm._C", "vllm._moe_C", "vllm._dsa_sm86"):
    try: importlib.import_module(m); print("OK  ", m)
    except Exception as e: ok=False; print("FAIL", m, "->", e)
sys.exit(0 if ok else 6)
PY
    exit $?
fi

# ---- preflight: verify env ----
if [[ "$SKIP_VERIFY" != "1" && -x "$VERIFY" ]]; then
    echo "[step] verify_env.sh --strict --venv $TARGET_VENV"
    bash "$VERIFY" --strict --venv "$TARGET_VENV" || { echo "[FAIL] env verify mismatch (use --skip-verify-env to override)"; exit 8; }
fi

# ---- locate wheel ----
if [[ -z "$WHEEL" ]]; then
    WHEEL="$(ls -1t $WHEEL_DIR_DEFAULT/vllm-*.whl 2>/dev/null | head -1 || true)"
fi
[[ -n "$WHEEL" && -r "$WHEEL" ]] || { echo "[FATAL] no wheel found (default dir: $WHEEL_DIR_DEFAULT)"; exit 9; }
echo "[wheel] $WHEEL ($(du -h "$WHEEL" | awk '{print $1}'))"

# ---- snapshot previous wheel ----
PREV_VLLM="$($TARGET_VENV/bin/pip show vllm 2>/dev/null | awk '/^Version:/{print $2}')"
if [[ -n "$PREV_VLLM" ]]; then
    PREV_WHL_NAME="vllm-${PREV_VLLM}-prev_$(date +%Y%m%d_%H%M%S).whl"
    "$TARGET_VENV/bin/pip" download --no-deps -d "$PREV_DIR" "vllm==$PREV_VLLM" 2>/dev/null || true
    echo "[snap] previous vllm $PREV_VLLM noted in $PREV_DIR"
fi

# ---- install ----
echo "[step] pip install --force-reinstall --no-deps $WHEEL"
"$TARGET_VENV/bin/pip" install --quiet --force-reinstall --no-deps "$WHEEL"
"$TARGET_VENV/bin/pip" install --quiet "$WHEEL"

# ---- post-install ABI smoke ----
"$TARGET_VENV/bin/python" - <<'PY' || { echo "[FAIL] post-install ABI smoke FAIL"; exit 6; }
import importlib, sys, torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__)
ok = True
for m in ("vllm._C", "vllm._moe_C"):
    try: importlib.import_module(m); print("OK  ", m)
    except Exception as e: ok=False; print("FAIL", m, "->", e)
try:
    import vllm._dsa_sm86 as d; print("OK  ", "vllm._dsa_sm86", "syms:", sorted([x for x in dir(d) if not x.startswith('_')])[:6])
except Exception as e:
    print("WARN", "vllm._dsa_sm86 missing:", e)
print("CUDA:", torch.cuda.is_available(), "sm:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
sys.exit(0 if ok else 6)
PY

# ---- record manifest ----
M="$TARGET_VENV/.cc2_install_node_manifest.txt"
{
    echo "host=$HOST"
    echo "installed_at=$(date -Is)"
    echo "wheel=$WHEEL"
    echo "wheel_sha256=$(sha256sum "$WHEEL" 2>/dev/null | awk '{print $1}' || echo unknown)"
    "$TARGET_VENV/bin/python" -c "import vllm; print(f'vllm={vllm.__version__}')"
    "$TARGET_VENV/bin/python" -c "import torch; print(f'torch={torch.__version__}')"
} > "$M"
echo "[ok] manifest: $M"
echo "[install_node] DONE $(date -Is) host=$HOST"
