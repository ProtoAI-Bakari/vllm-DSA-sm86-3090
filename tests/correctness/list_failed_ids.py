#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK gate-output utility; not a perf bench
# Extract the prompt ids that disagreed between baseline + under-test from a
# verify_gate.py --json-out file. Useful for CC9 to re-run only the failing
# subset against a candidate fix instead of re-grading all 1500.
#
# Run:
#   python3 list_failed_ids.py <CC6_GATE_RESULTS_<integ>.json>
#   python3 list_failed_ids.py <json> --jsonl-out failed_ids.jsonl
#   python3 list_failed_ids.py <json> --replay-prompts prompts.jsonl --out replay.jsonl
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_in", help="verify_gate.py --json-out file")
    ap.add_argument("--replay-prompts", default=None,
                    help="if given, write a subset of prompts.jsonl containing only the failed ids")
    ap.add_argument("--out", default=None, help="output path; default stdout")
    args = ap.parse_args()

    with open(args.json_in) as f:
        rep = json.load(f)
    failed = [d["id"] for d in rep.get("sample_disagreements", [])]
    # Add any ids missing from common as also-disagreement (capture skew)
    failed = sorted(set(failed))

    out_lines = []
    if args.replay_prompts:
        with open(args.replay_prompts) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if o.get("id") in set(failed):
                    out_lines.append(line)
    else:
        for fid in failed:
            out_lines.append(fid)

    out = "\n".join(out_lines) + "\n" if out_lines else ""
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
        print(f"wrote {len(failed)} failed ids → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
