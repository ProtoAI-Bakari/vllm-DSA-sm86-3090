#!/usr/bin/env bash
# install_dsa_sm86.sh — CC2 L4-cycle hook installer.
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC2]
#
# Wires into the L4 cycle workflow BETWEEN mirror-update and launch:
#
#   L4 step               | hook
#   ----------------------+----------------------------------------------
#   1. drain cluster      | sky_drain.sh
#   2. git mirror update  | mirror_dsa_repo_to_cluster.sh
# >>3. INSTALL csrc/wheel | install_dsa_sm86.sh   <<--- this script
#   4. launch             | sky_launch.sh
#
# Idempotent. Reads BUILD_INFO from /repo/INSTALLERS/vllm.dsa.sm86/BUILD_INFO when present.
#
# Use:
#   bash scripts/install_dsa_sm86.sh                    # pull + rebuild if vendor SHA advanced
#   bash scripts/install_dsa_sm86.sh --force            # always rebuild
#   bash scripts/install_dsa_sm86.sh --abi-only         # only run ABI smoke vs current wheel
#   bash scripts/install_dsa_sm86.sh --target-venv DIR  # override target venv

set -uo pipefail

REPO_BARE="${REPO_BARE:-/repo/INSTALLERS/vllm-DSA-sm86-3090.git}"
CHECKOUT_DIR="${CHECKOUT_DIR:-/repo/INSTALLERS/vllm-DSA-sm86-3090}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${CHECKOUT_DIR}/scripts/build_vllm_dsa_sm86.sh}"
TARGET_VENV="${TARGET_VENV:-/home/z/.venv-vllm_pr40760_shaaa114601_torch2.10.0_cu128_ray2.54.0_py3.12_dsv4}"
PROFILE="${PROFILE:-dsv4-pr40760}"
LOG_DIR="${LOG_DIR:-${HOME}/INSTALL_LOGS}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/install_dsa_sm86_$(date +%Y%m%d_%H%M%S).log}"
SHA_FILE="${SHA_FILE:-${TARGET_VENV}/.cc2_installed_oss_sha}"

FORCE=0
ABI_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --abi-only) ABI_ONLY=1 ;;
        --target-venv) shift; TARGET_VENV="$1" ;;
        --target-venv=*) TARGET_VENV="${arg#--target-venv=}" ;;
        --help|-h) grep -E '^# ' "$0" | head -25; exit 0 ;;
    esac
done

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[install] start $(date -Is) host=$(hostname) target=$TARGET_VENV profile=$PROFILE"

# ---- step 1: refresh checkout from bare mirror ----
if [[ ! -d "$CHECKOUT_DIR/.git" ]]; then
    echo "[step] clone bare mirror → $CHECKOUT_DIR"
    git clone "$REPO_BARE" "$CHECKOUT_DIR"
fi
( cd "$CHECKOUT_DIR" && git fetch --all --prune 2>&1 | tail -5 || true )
# Pin to whichever lane has the latest CC2 build script (lane-cc2-build).
( cd "$CHECKOUT_DIR" && git checkout lane-cc2-build 2>/dev/null || git checkout main )
( cd "$CHECKOUT_DIR" && git pull --ff-only 2>&1 | tail -5 || true )
NEW_OSS_SHA="$(git -C "$CHECKOUT_DIR" rev-parse HEAD)"
LAST_INSTALLED="$(cat "$SHA_FILE" 2>/dev/null || echo none)"
echo "[info] oss_sha new=$NEW_OSS_SHA last_installed=$LAST_INSTALLED"

if [[ "$ABI_ONLY" == "1" ]]; then
    bash "$BUILD_SCRIPT" --profile "$PROFILE" --abi-only
    exit $?
fi

if [[ "$NEW_OSS_SHA" == "$LAST_INSTALLED" && "$FORCE" != "1" ]]; then
    echo "[ok] no oss diff since last install ($NEW_OSS_SHA) — skipping rebuild (use --force to override)"
    exit 0
fi

# ---- step 2: drive build script ----
echo "[step] invoking $BUILD_SCRIPT --profile $PROFILE"
EXTRA_ARGS=()
[[ "$FORCE" == "1" ]] && EXTRA_ARGS+=(--rebuild)
VENV_DIR="$TARGET_VENV" PROFILE="$PROFILE" bash "$BUILD_SCRIPT" --profile "$PROFILE" "${EXTRA_ARGS[@]}"
RC=$?
if [[ "$RC" == "0" ]]; then
    mkdir -p "$(dirname "$SHA_FILE")"
    echo "$NEW_OSS_SHA" > "$SHA_FILE"
    echo "[ok] install COMPLETE oss_sha=$NEW_OSS_SHA recorded at $SHA_FILE"
else
    echo "[FAIL] build returned rc=$RC"
    exit "$RC"
fi
