#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK capture summarizer; not a perf bench
# Quick stats over a baseline-format JSONL: error rate, completion-token
# distribution, mean TTFT/TPOT/TG_TPS, per-category breakdown. Useful for CC0
# dashboards + CC9 capture-quality sanity check before grading.
#
# Run:
#   python3 summarize_captures.py <baseline.jsonl>
#   python3 summarize_captures.py <baseline.jsonl> --json
import argparse, json, statistics, sys
from collections import defaultdict


def median(xs):
    return round(statistics.median(xs), 4) if xs else None


def p_pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return round(s[int(min(len(s) - 1, max(0, p * (len(s) - 1) / 100)))], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    n = 0
    n_err = 0
    by_cat = defaultdict(lambda: {"n": 0, "errors": 0})
    ttfts, tpots, tg_tpss, walls, completion_toks = [], [], [], [], []

    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            cat = rec.get("category", "unknown")
            by_cat[cat]["n"] += 1
            if "error" in rec:
                n_err += 1
                by_cat[cat]["errors"] += 1
                continue
            m = rec.get("metrics") or {}
            for k, lst in (("TTFT_s", ttfts), ("TPOT_s", tpots), ("TG_TPS", tg_tpss),
                           ("wall_s", walls), ("completion_tokens", completion_toks)):
                v = m.get(k)
                if isinstance(v, (int, float)):
                    lst.append(float(v))

    summary = {
        "path": args.path,
        "n_records": n,
        "n_errors": n_err,
        "error_rate": round(n_err / n, 4) if n else 0,
        "metrics": {
            "TTFT_s":  {"n": len(ttfts), "p50": median(ttfts), "p95": p_pct(ttfts, 95)},
            "TPOT_s":  {"n": len(tpots), "p50": median(tpots), "p95": p_pct(tpots, 95)},
            "TG_TPS":  {"n": len(tg_tpss), "p50": median(tg_tpss), "p95": p_pct(tg_tpss, 95), "max": max(tg_tpss) if tg_tpss else None},
            "wall_s":  {"n": len(walls), "p50": median(walls), "p95": p_pct(walls, 95)},
            "completion_tokens": {"n": len(completion_toks), "mean": round(statistics.mean(completion_toks), 1) if completion_toks else None,
                                  "p50": median(completion_toks), "max": max(completion_toks) if completion_toks else None},
        },
        "per_category": {c: dict(by_cat[c]) for c in sorted(by_cat)},
    }

    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"# Capture summary — {args.path}")
        print(f"records={n}  errors={n_err}  error_rate={summary['error_rate']:.4f}")
        print()
        print("Latency:")
        for k in ("TTFT_s", "TPOT_s", "wall_s"):
            d = summary["metrics"][k]
            print(f"  {k:10s} n={d['n']:5d}  p50={d['p50']}  p95={d['p95']}")
        d = summary["metrics"]["TG_TPS"]
        print(f"  TG_TPS     n={d['n']:5d}  p50={d['p50']}  p95={d['p95']}  max={d['max']}")
        d = summary["metrics"]["completion_tokens"]
        print(f"  out_toks   n={d['n']:5d}  p50={d['p50']}  mean={d['mean']}  max={d['max']}")
        print()
        print("Per-category:")
        for c, d in summary["per_category"].items():
            print(f"  {c:20s} n={d['n']:5d}  err={d['errors']:5d}")


if __name__ == "__main__":
    main()
