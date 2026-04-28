#!/usr/bin/env python3
# --ProtoAI-Bakari--
# tier_classify.py — Story #7. Classify bench output into Bronze/Silver/Gold/Platinum/Diamond/Cosmic
# on both aggregate and conc=1 scales (per corpus §8.1).
#
# Usage:
#   tier_classify.py --conc-sweep <conc_sweep.jsonl>
#   tier_classify.py --agg-tps 314 --conc1-tps 28
#
# Returns JSON with: aggregate_tier, conc1_tier, agg_tps, conc1_tps, target_gaps.

from __future__ import annotations
import argparse, json, sys
from typing import Optional


AGG_TIERS = [
    ("cosmic",   1000),
    ("diamond",   900),
    ("platinum",  700),
    ("gold",      500),
    ("silver",    300),
    ("bronze",    100),
]

CONC1_TIERS = [
    ("diamond",  75),
    ("platinum", 65),
    ("gold",     55),
    ("silver",   35),
    ("bronze",   25),
]


def classify(value: float, ladder: list[tuple[str, float]]) -> dict:
    for name, threshold in ladder:
        if value >= threshold:
            higher = next((n for n, t in ladder if t > threshold), None)
            higher_th = next((t for n, t in ladder if t > threshold), None)
            return {
                "tier": name,
                "value": value,
                "threshold_met": threshold,
                "next_tier": higher,
                "next_threshold": higher_th,
                "gap_to_next": (higher_th - value) if higher_th else None,
            }
    return {
        "tier": "below_bronze",
        "value": value,
        "threshold_met": 0,
        "next_tier": ladder[-1][0],
        "next_threshold": ladder[-1][1],
        "gap_to_next": ladder[-1][1] - value,
    }


def parse_conc_sweep(path: str) -> dict:
    """Read a conc_sweep.sh summary jsonl. Each line records one conc step.
    Compute: best agg_tps across the sweep, conc=1 tps if present."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Each row references a "raw" jsonl file; aggregate from per-conc raw if available.
    best_agg = 0.0
    conc1_tps: Optional[float] = None
    by_conc: dict[int, dict] = {}
    for row in rows:
        conc = int(row.get("conc", 0))
        raw_path = row.get("raw")
        if not raw_path: continue
        try:
            raw_rows = [json.loads(l) for l in open(raw_path) if l.strip()]
        except FileNotFoundError:
            continue
        ok = [r for r in raw_rows if r.get("ok")]
        if not ok: continue
        total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
        wall = max(row.get("step_wall_s", 1), 1e-6)
        agg = total_tokens / wall
        per_req_mean = sum(r.get("tps_per_request", 0) for r in ok) / max(len(ok), 1)
        by_conc[conc] = {"agg_tps": agg, "per_request_tps_mean": per_req_mean,
                          "n_ok": len(ok), "wall_s": wall}
        best_agg = max(best_agg, agg)
        if conc == 1:
            conc1_tps = per_req_mean

    return {"best_agg_tps": best_agg, "conc1_tps": conc1_tps, "by_conc": by_conc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conc-sweep", help="Path to conc_sweep.sh summary jsonl")
    p.add_argument("--agg-tps", type=float, help="Override: aggregate tokens/sec")
    p.add_argument("--conc1-tps", type=float, help="Override: conc=1 tokens/sec")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.conc_sweep:
        parsed = parse_conc_sweep(args.conc_sweep)
        agg_tps = args.agg_tps if args.agg_tps is not None else parsed["best_agg_tps"]
        conc1_tps = args.conc1_tps if args.conc1_tps is not None else parsed["conc1_tps"]
        by_conc = parsed["by_conc"]
    else:
        agg_tps = args.agg_tps or 0
        conc1_tps = args.conc1_tps or 0
        by_conc = {}

    agg_class = classify(agg_tps, AGG_TIERS)
    conc1_class = classify(conc1_tps if conc1_tps is not None else 0, CONC1_TIERS)

    out = {
        "aggregate": agg_class,
        "conc1": conc1_class,
        "by_conc": by_conc,
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
