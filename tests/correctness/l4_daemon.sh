#!/bin/bash
# --ProtoAI-Bakari--
# Story 9 daemon: polls bridge.db for topic=l4_capture_ready posts, dispatches
# run_l4_verdict.sh on each, marks the request claimed (so it isn't re-fired).
#
# Bridge body schema expected from CC9:
#   "integ=<label> under_test=<path> [baseline=<path>]"
# Examples:
#   integ=cc3-rev2 under_test=/Users/z/AGENT/comms/captures/cc3-rev2.jsonl
#   integ=cc4-rev1 under_test=/repo/.../under_test.jsonl baseline=/repo/.../baseline.jsonl
#
# Run:
#   ./l4_daemon.sh                         # default 30s poll
#   POLL_INTERVAL=60 ./l4_daemon.sh        # slower poll
#   ./l4_daemon.sh --once                  # single pass, exit
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:-${HOME}/AGENT/comms/cc6_l4_state}"
mkdir -p "$STATE_DIR"
INTERVAL="${POLL_INTERVAL:-30}"
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

# Bridge query: read recent l4_capture_ready posts (last 200 entries scanned)
poll() {
  python3 - <<'PY'
import json, os, subprocess, sys
state = os.path.expanduser(os.environ.get("STATE_DIR", "~/AGENT/comms/cc6_l4_state"))
seen = set()
seen_path = os.path.join(state, "claimed.txt")
if os.path.exists(seen_path):
    with open(seen_path) as f:
        seen = {ln.strip() for ln in f if ln.strip()}
out = subprocess.run(
    ["python3", os.path.expanduser("~/AGENT/comms/bridge.py"), "read", "--topic", "l4_capture_ready", "--limit", "50"],
    capture_output=True, text=True,
)
for line in out.stdout.splitlines():
    line = line.strip()
    if not line:
        continue
    # bridge.py read format: "[ts] sender #topic: body"
    body_idx = line.find(": ")
    if body_idx < 0:
        continue
    head = line[:body_idx]
    body = line[body_idx+2:]
    key = head.split("]", 1)[0] + "|" + body[:80]
    if key in seen:
        continue
    fields = {}
    for tok in body.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v
    if "integ" not in fields or "under_test" not in fields:
        sys.stderr.write(f"  [skip malformed] {body[:120]}\n")
        seen.add(key)
        continue
    print(json.dumps({"key": key, "integ": fields["integ"],
                      "under_test": fields["under_test"],
                      "baseline": fields.get("baseline", "")}))
    seen.add(key)
with open(seen_path, "w") as f:
    for k in sorted(seen):
        f.write(k + "\n")
PY
}

while :; do
  echo "[l4_daemon] poll @ $(date)"
  poll | while IFS= read -r task; do
    [[ -z "$task" ]] && continue
    INTEG=$(echo "$task" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["integ"])')
    UNDER=$(echo "$task" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["under_test"])')
    BASE=$(echo "$task" | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["baseline"])')
    echo "[l4_daemon] dispatching integ=$INTEG under=$UNDER base=${BASE:-default}"
    if [[ -n "$BASE" ]]; then
      bash "$HERE/run_l4_verdict.sh" --integ "$INTEG" --under-test "$UNDER" --baseline "$BASE" || true
    else
      bash "$HERE/run_l4_verdict.sh" --integ "$INTEG" --under-test "$UNDER" || true
    fi
  done
  [[ "$ONCE" == "1" ]] && break
  sleep "$INTERVAL"   # SLOW_OP polling
done
