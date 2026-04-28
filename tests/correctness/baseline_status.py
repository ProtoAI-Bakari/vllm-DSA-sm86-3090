#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK gate-readiness probe; not a perf bench
# Quick health check: is baseline.jsonl present, complete, recent enough that
# the gate is bootable RIGHT NOW? Used by cycle5_smoke.sh + CC0 cron + CC9
# pre-flight to avoid firing the gate against a stale or missing baseline.
#
# Run:
#   python3 baseline_status.py [--baseline path] [--max-age-hours 168]
#
# Exit: 0 healthy / 1 stale / 2 missing or empty / 3 high error rate
import argparse, json, os, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(Path(__file__).parent / "baseline.jsonl"))
    ap.add_argument("--expected-count", type=int, default=1500)
    ap.add_argument("--max-age-hours", type=float, default=168.0,  # 1 week
                    help="exit 1 if baseline mtime older than this")
    ap.add_argument("--max-error-pct", type=float, default=2.0,
                    help="exit 3 if more than this fraction of records errored")
    args = ap.parse_args()

    p = Path(args.baseline)
    if not p.exists():
        print(f"MISSING path={p}", file=sys.stderr)
        sys.exit(2)
    if p.stat().st_size == 0:
        print(f"EMPTY path={p}", file=sys.stderr)
        sys.exit(2)

    age_h = (time.time() - p.stat().st_mtime) / 3600.0
    n = 0
    n_err = 0
    cats = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            cats[rec.get("category", "?")] = cats.get(rec.get("category", "?"), 0) + 1
            if "error" in rec:
                n_err += 1

    err_pct = (n_err / n * 100) if n else 0
    state = {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "age_hours": round(age_h, 2),
        "n_records": n,
        "n_expected": args.expected_count,
        "completeness_pct": round(n / args.expected_count * 100, 2) if args.expected_count else None,
        "n_errors": n_err,
        "error_pct": round(err_pct, 3),
        "categories": cats,
    }
    print(json.dumps(state, indent=2))

    if age_h > args.max_age_hours:
        print(f"STALE — age={age_h:.1f}h > max={args.max_age_hours}h", file=sys.stderr)
        sys.exit(1)
    if err_pct > args.max_error_pct:
        print(f"HIGH_ERRORS — error_pct={err_pct:.2f}% > max={args.max_error_pct}%", file=sys.stderr)
        sys.exit(3)
    if n < args.expected_count:
        print(f"PARTIAL — {n}/{args.expected_count} records ({state['completeness_pct']}%)", file=sys.stderr)
        # Partial baseline is usable but warn
    print("HEALTHY", file=sys.stderr)


if __name__ == "__main__":
    main()
