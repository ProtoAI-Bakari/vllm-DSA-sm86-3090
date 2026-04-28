#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK status digest tool; not a perf bench
# Story W3-NEW: CC6 daily status digest. Reads bridge.db for CC6 posts in the
# last 24h, classifies, summarizes for CC0's 3-min cron. Saves CC0 from
# parsing raw bridge entries every poll.
#
# Output: ~/AGENT/comms/CC6_DAILY_STATUS.md (also stdout) + bridge daily_status.
#
# Run:
#   python3 cc6_daily_status.py                # last 24h
#   python3 cc6_daily_status.py --hours 6      # last 6h
#   python3 cc6_daily_status.py --no-bridge
import argparse, json, os, re, subprocess, sys, time
from collections import defaultdict, Counter
from pathlib import Path

BRIDGE = os.path.expanduser("~/AGENT/comms/bridge.py")


def read_bridge_topic(topic, limit=200):
    out = subprocess.run(
        ["python3", BRIDGE, "read", "--topic", topic, "--limit", str(limit)],
        capture_output=True, text=True, timeout=15,
    )
    return out.stdout.splitlines()


# bridge.py read line format: "[YYYY-MM-DDTHH:MM:SSZ] <sender> #<topic>: <body>"
LINE_RE = re.compile(r"^\[([^\]]+)\]\s+(\S+)\s+#(\S+):\s*(.*)$")


def parse_lines(lines, since_ts):
    out = []
    for ln in lines:
        m = LINE_RE.match(ln)
        if not m:
            continue
        ts_s, sender, topic, body = m.groups()
        try:
            tt = time.strptime(ts_s.split("+")[0].split(".")[0].rstrip("Z"), "%Y-%m-%dT%H:%M:%S")
            tt_ts = int(time.mktime(tt))
        except Exception:
            tt_ts = int(time.time())
        if tt_ts < since_ts:
            continue
        out.append({"ts": ts_s, "ts_unix": tt_ts, "sender": sender, "topic": topic, "body": body})
    return out


def collect(hours):
    since = int(time.time() - hours * 3600)
    topics = ["story_complete", "milestone", "l4_result", "l4_shipped", "tier_result",
              "grill_result", "longctx_result", "endpoint_health", "blocker", "warning",
              "backlog_pruned", "backlog_done", "ack", "p0_pace_ack", "boot"]
    all_entries = []
    for t in topics:
        try:
            lines = read_bridge_topic(t, limit=300)
            for e in parse_lines(lines, since):
                if e["sender"] == "claude-cc6":
                    all_entries.append(e)
        except Exception as ex:
            print(f"  WARN: bridge read {t} failed: {ex}", file=sys.stderr)
    all_entries.sort(key=lambda e: e["ts_unix"])
    return all_entries


def summarize(entries):
    by_topic = defaultdict(list)
    for e in entries:
        by_topic[e["topic"]].append(e)

    stories = by_topic.get("story_complete", [])
    l4_results = by_topic.get("l4_result", []) + by_topic.get("l4_shipped", [])
    pass_count = sum(1 for e in l4_results if "PASS" in e["body"])
    fail_count = sum(1 for e in l4_results if "FAIL" in e["body"])

    blockers = [e for e in by_topic.get("blocker", []) if "RESOLVED" not in e["body"].upper()]
    warnings = by_topic.get("warning", [])

    endpoint = by_topic.get("endpoint_health", [])
    last_endpoint = endpoint[-1] if endpoint else None

    tier = by_topic.get("tier_result", [])
    last_tier = tier[-1] if tier else None

    return {
        "n_stories_shipped": len(stories),
        "stories_titles": [e["body"].split(".")[0][:120] for e in stories[-15:]],
        "n_l4_runs": len(l4_results),
        "l4_pass": pass_count,
        "l4_fail": fail_count,
        "n_blockers": len(blockers),
        "blocker_summaries": [e["body"][:140] for e in blockers[-5:]],
        "n_warnings": len(warnings),
        "warning_summaries": [e["body"][:140] for e in warnings[-3:]],
        "last_endpoint_health": last_endpoint["body"] if last_endpoint else None,
        "last_tier": last_tier["body"] if last_tier else None,
        "first_ts": entries[0]["ts"] if entries else None,
        "last_ts": entries[-1]["ts"] if entries else None,
    }


def render(summary, hours):
    L = []
    L.append(f"# CC6 Daily Status — last {hours}h")
    L.append("")
    L.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append(f"**Window:** {summary['first_ts']} → {summary['last_ts']}")
    L.append("")
    L.append("## Headlines")
    L.append(f"- Stories shipped:  **{summary['n_stories_shipped']}**")
    L.append(f"- L4 runs:          **{summary['n_l4_runs']}** (PASS {summary['l4_pass']} / FAIL {summary['l4_fail']})")
    L.append(f"- Active blockers:  **{summary['n_blockers']}**")
    L.append(f"- Warnings (3):     {summary['n_warnings']}")
    if summary["last_endpoint_health"]:
        L.append(f"- Endpoint:         {summary['last_endpoint_health']}")
    if summary["last_tier"]:
        L.append(f"- Last tier verdict: {summary['last_tier']}")
    L.append("")
    if summary["stories_titles"]:
        L.append("## Recent ships")
        for t in summary["stories_titles"]:
            L.append(f"- {t}")
        L.append("")
    if summary["blocker_summaries"]:
        L.append("## Active blockers")
        for b in summary["blocker_summaries"]:
            L.append(f"- {b}")
        L.append("")
    if summary["warning_summaries"]:
        L.append("## Warnings")
        for w in summary["warning_summaries"]:
            L.append(f"- {w}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-bridge", action="store_true")
    args = ap.parse_args()
    entries = collect(args.hours)
    summary = summarize(entries)
    md = render(summary, args.hours)
    out = args.report or os.path.expanduser("~/AGENT/comms/CC6_DAILY_STATUS.md")
    with open(out, "w") as f:
        f.write(md)
    sys.stdout.write(md)
    print(f"\n[cc6_daily_status] {summary['n_stories_shipped']} ships / {summary['n_l4_runs']} L4 / {summary['n_blockers']} blockers → {out}", file=sys.stderr)
    if not args.no_bridge:
        body = (f"window={args.hours}h ships={summary['n_stories_shipped']} "
                f"l4={summary['l4_pass']}P/{summary['l4_fail']}F "
                f"blockers={summary['n_blockers']} report={out}")
        try:
            subprocess.run(["python3", BRIDGE, "post", "--from", "claude-cc6",
                           "--topic", "daily_status", "--body", body], check=False)
        except Exception:
            pass


if __name__ == "__main__":
    main()
