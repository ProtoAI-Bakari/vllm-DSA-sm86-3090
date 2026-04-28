#!/usr/bin/env python3
# --ProtoAI-Bakari--
# DRY bridge poster used by every gate script. Wraps subprocess invocation of
# ~/AGENT/comms/bridge.py with consistent retry + timeout + same agent id +
# graceful failure (gate scripts never crash on bridge unavailability).
import os
import shutil
import subprocess
import sys
from typing import Optional

BRIDGE = os.path.expanduser("~/AGENT/comms/bridge.py")
DEFAULT_AGENT = "claude-cc6"


def post(topic: str, body: str, *,
         agent: str = DEFAULT_AGENT,
         timeout_s: float = 10.0,
         retry: int = 2,
         silent: bool = False) -> bool:
    """Post one bridge message; True on success, False on failure (logged)."""
    if not os.path.exists(BRIDGE):
        if not silent:
            print(f"[bridge_helper] bridge.py not found at {BRIDGE}", file=sys.stderr)
        return False
    cmd = ["python3", BRIDGE, "post", "--from", agent, "--topic", topic, "--body", body]
    last_err = None
    for attempt in range(retry + 1):
        try:
            r = subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True)
            if r.returncode == 0:
                return True
            last_err = (r.returncode, r.stderr.strip()[:200])
        except subprocess.TimeoutExpired:
            last_err = ("timeout", f"{timeout_s}s")
        except Exception as e:
            last_err = ("exception", f"{type(e).__name__}: {e}")
    if not silent:
        print(f"[bridge_helper] post failed after {retry+1} tries: {last_err}", file=sys.stderr)
    return False


def post_l4_result(integ: str, verdict: str, **fields) -> bool:
    """Convenience for the L4 verdict post — verdict is PASS|FAIL|ERROR-N."""
    parts = [verdict, f"integ={integ}"] + [f"{k}={v}" for k, v in fields.items()]
    return post("l4_result", " ".join(parts))


def post_story_complete(sha: str, path: str, note: str = "") -> bool:
    body = f"{sha} {path}"
    if note:
        body += f" {note}"
    return post("story_complete", body)


def post_milestone(body: str) -> bool:
    return post("milestone", body)


def post_blocker(body: str) -> bool:
    return post("blocker", body)


# CLI shim so bash scripts can use this module without importing from python:
#   python3 -m bridge_helper milestone "ship 6f1fcd8 lane-cc6-gate"
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: bridge_helper.py <topic> <body...>", file=sys.stderr)
        sys.exit(2)
    topic = sys.argv[1]
    body = " ".join(sys.argv[2:])
    sys.exit(0 if post(topic, body) else 1)
