#!/bin/bash
# --ProtoAI-Bakari--
# CC0 22:57Z (4): launch l4_daemon.sh inside an EXISTING tmux pane (no new
# session — block_new_tmux_session_hook). Default pane = WC6 (CC6's only allowed
# launch pane). Verifies daemon polls + reacts to a TEST l4_capture_ready post
# without touching real CC9 captures.
#
# Run:
#   ./run_l4_daemon_in_pane.sh                          # launch in WC6
#   PANE=WC6 ./run_l4_daemon_in_pane.sh
#   ./run_l4_daemon_in_pane.sh --self-test              # inject a test post + verify pickup
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PANE="${PANE:-WC6}"
SELF_TEST=0
[[ "${1:-}" == "--self-test" ]] && SELF_TEST=1

if ! tmux has-session -t "$PANE" 2>/dev/null; then
  echo "[run_l4_daemon_in_pane] tmux session $PANE missing — abort" >&2
  exit 2
fi

if [[ "$SELF_TEST" == "1" ]]; then
  # Self-test mode: emit a TEST l4_capture_ready post (under_test path doesn't exist)
  # and capture daemon's response from its bridge l4_result reply.
  TEST_INTEG="cc6-self-test-$(date +%s)"
  TEST_PATH="/tmp/cc6_self_test_capture_${TEST_INTEG}.jsonl"
  cp "$HERE/../../tests/correctness/baseline.jsonl" "$TEST_PATH" 2>/dev/null || \
    cp "$HERE/baseline.jsonl" "$TEST_PATH"
  echo "[run_l4_daemon_in_pane] self-test: integ=$TEST_INTEG capture=$TEST_PATH"
  python3 "${HOME}/AGENT/comms/bridge.py" post --from claude-cc6 --topic l4_capture_ready \
    --body "integ=${TEST_INTEG} under_test=${TEST_PATH} baseline=${HERE}/baseline.jsonl"
  echo "[run_l4_daemon_in_pane] posted; daemon will pick up on next poll cycle (≤30s)"
  echo "[run_l4_daemon_in_pane] verify: python3 ~/AGENT/comms/bridge.py read --topic l4_result --limit 5"
  exit 0
fi

# Normal launch — send the daemon command into PANE
CMD="bash $HERE/l4_daemon.sh"
echo "[run_l4_daemon_in_pane] launching daemon in pane=$PANE"
echo "[run_l4_daemon_in_pane] cmd=$CMD"
tmux send-keys -t "$PANE" "$CMD" Enter
echo "[run_l4_daemon_in_pane] sent; tail with: tmux capture-pane -t $PANE -p -S -50"
