#!/usr/bin/env bash
# build_matrix.sh — CC2 Story 6: dual-track 0.19 / 0.20 build matrix.
# Apply same P6/P7/P8 patches against both vllm versions, identify build deltas.
# Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC2]
#
# Use:
#   bash scripts/build_matrix.sh                  # build both 0.19.1+pr40760 and 0.20.0
#   bash scripts/build_matrix.sh --only 0.20.0
#   bash scripts/build_matrix.sh --rebuild
#   bash scripts/build_matrix.sh --report-only    # diff existing manifests
#
# Output: build_matrix_report.md with build success / wall / wheel size / smoke / patch outcome
# per (vllm_version, sha) tuple.

set -uo pipefail

ROOT="${ROOT:-/repo/INSTALLERS}"
WHEELS_BASE="${WHEELS_BASE:-/repo/wheels}"
LOG_DIR="${LOG_DIR:-${HOME}/INSTALL_LOGS}"
REPORT="${REPORT:-${LOG_DIR}/build_matrix_report_$(date +%Y%m%d_%H%M%S).md}"
SCRIPT_SELF="$(cd "$(dirname "$0")" && pwd)/build_vllm_dsa_sm86.sh"

# (version_tag, sha, src_subdir, wheel_subdir, venv_name)
MATRIX=(
    "0.20.0|101584af0|vllm.dsa.sm86|vllm.dsa.sm86|.venv-vllm_dsa_sm86_baseline"
    "0.19.1+pr40760|aa114601d|vllm.dsa.sm86_pr40760|vllm.dsa.sm86_pr40760|.venv-vllm_dsa_sm86_pr40760"
)

ONLY=""
REBUILD=0
REPORT_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --only) shift; ONLY="$1" ;;
        --only=*) ONLY="${arg#--only=}" ;;
        --rebuild) REBUILD=1 ;;
        --report-only) REPORT_ONLY=1 ;;
        --help|-h) grep -E '^# ' "$0" | head -25; exit 0 ;;
    esac
done

mkdir -p "$LOG_DIR"
echo "# build_matrix report — $(date -Is)" > "$REPORT"
echo "" >> "$REPORT"
echo "| version | sha | src | wheel | smoke | wall(s) | wheel_size | patches | notes |" >> "$REPORT"
echo "|---|---|---|---|---|---|---|---|---|" >> "$REPORT"

run_one() {
    local row="$1"
    IFS='|' read -r VER SHA SRCSUB WHLSUB VENV <<< "$row"
    [[ -n "$ONLY" && "$ONLY" != "$VER" ]] && return 0

    local SRC="$ROOT/$SRCSUB/vllm-source"
    local WHL="$WHEELS_BASE/$WHLSUB"
    local VDIR="${HOME}/.venvs/$VENV"
    local LOG="$LOG_DIR/build_matrix_${VER//[+\/]/_}_$(date +%Y%m%d_%H%M%S).log"
    local NOTES=""

    if [[ "$REPORT_ONLY" == "1" ]]; then
        local M="$VDIR/.cc2_build_manifest.txt"
        local sm="?"
        local wsz="?"
        if [[ -r "$M" ]]; then
            sm="OK"
            wsz="$(ls -lh "$WHL"/vllm-*.whl 2>/dev/null | awk 'NR==1{print $5}' || echo '?')"
            NOTES="manifest:$(stat -f %m "$M" 2>/dev/null || stat -c %Y "$M" 2>/dev/null)"
        else
            sm="missing"
            NOTES="no manifest"
        fi
        printf "| %s | %s | %s | %s | %s | - | %s | - | %s |\n" \
            "$VER" "$SHA" "$SRC" "$WHL" "$sm" "$wsz" "$NOTES" >> "$REPORT"
        return 0
    fi

    echo "[matrix] === $VER (SHA $SHA) ===" | tee -a "$LOG"
    local T0=$(date +%s)
    SHA_TARGET="$SHA" \
    SRC_DIR="$SRC" \
    WHEEL_DIR="$WHL" \
    VENV_NAME="$VENV" \
    VENV_DIR="$VDIR" \
    LOG_FILE="$LOG" \
        bash "$SCRIPT_SELF" $([[ "$REBUILD" == "1" ]] && echo --rebuild) >>"$LOG" 2>&1
    local rc=$?
    local T1=$(date +%s)
    local wall=$((T1 - T0))
    local wsz="-"
    [[ -d "$WHL" ]] && wsz="$(ls -lh "$WHL"/vllm-*.whl 2>/dev/null | awk 'NR==1{print $5}' || echo '?')"
    local smoke="FAIL"
    [[ "$rc" == "0" ]] && smoke="OK"
    [[ -r "$VDIR/.cc2_build_manifest.txt" ]] && NOTES="manifest_present"
    printf "| %s | %s | %s | %s | %s | %s | %s | P6+P7+P8 | rc=%s %s |\n" \
        "$VER" "$SHA" "$SRC" "$WHL" "$smoke" "$wall" "$wsz" "$rc" "$NOTES" >> "$REPORT"
}

for row in "${MATRIX[@]}"; do
    run_one "$row"
done

echo "" >> "$REPORT"
echo "## Delta analysis" >> "$REPORT"
echo "" >> "$REPORT"
echo "* If 0.20.0 baseline FAILs but 0.19.1+pr40760 passes → route DSV4 path through 0.19.1 fork." >> "$REPORT"
echo "* If both pass → confirm 0.20.0 is preferred (newer flashinfer + mistral_common). Diff manifests for transitive package skew." >> "$REPORT"
echo "* If both fail → escalate to CC0 / CCX research lane (root cause is environmental, not version-specific)." >> "$REPORT"

echo "[matrix] DONE  report: $REPORT"
