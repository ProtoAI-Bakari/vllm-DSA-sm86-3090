#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK perf-tier classifier; consumes a CC8 bench JSON and emits a tier
# verdict against LEAD corpus PART2 §8.1 thresholds.
#
# Aggregate t/s tiers:  Bronze 100 / Silver 300 / Gold 500 / Platinum 700 / Diamond 900 / Cosmic 1000
# Conc=1 t/s tiers:     Bronze  25 / Silver  35 / Gold  55 / Platinum  65 / Diamond  75
#
# Input JSON schema (best-effort flexible):
#   {
#     "model": "...", "topology": "...",
#     "results": [
#        {"conc": 1, "tps_per_req": <float>, "tps_aggregate": <float>, ...},
#        {"conc": 4, "tps_per_req": <float>, "tps_aggregate": <float>, ...},
#        ...
#     ]
#   }
# OR top-level list of result dicts.
#
# Output:
#   - markdown report ~/AGENT/comms/CC6_TIER_<integ>.md
#   - bridge topic=tier_result body=<best-tier> aggregate_tps=<...> conc1_tps=<...>
#   - exit 0 on Bronze+, 1 on below-Bronze
import argparse, json, os, sys, time

AGG_TIERS = [
    ("Cosmic",   1000),
    ("Diamond",   900),
    ("Platinum",  700),
    ("Gold",      500),
    ("Silver",    300),
    ("Bronze",    100),
]
CONC1_TIERS = [
    ("Diamond",  75),
    ("Platinum", 65),
    ("Gold",     55),
    ("Silver",   35),
    ("Bronze",   25),
]


def classify(value, tiers):
    if value is None:
        return ("UNKNOWN", None)
    for name, thr in tiers:
        if value >= thr:
            return (name, thr)
    return ("below-Bronze", None)


def load(path):
    if path == "-":
        return json.load(sys.stdin)
    with open(path) as f:
        return json.load(f)


def extract_results(blob):
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for k in ("results", "bench", "measurements", "runs"):
            if isinstance(blob.get(k), list):
                return blob[k]
    return []


def best(results, key):
    vals = []
    for r in results:
        if not isinstance(r, dict):
            continue
        v = r.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return max(vals) if vals else None


def conc1_per_req(results):
    """If conc==1 row exists, use its tps_per_req; else None."""
    for r in results:
        if isinstance(r, dict) and r.get("conc") == 1:
            v = r.get("tps_per_req") or r.get("tps_aggregate")
            if isinstance(v, (int, float)):
                return float(v)
    return None


def render(integ, model, topology, agg_max, conc1_max, agg_tier, c1_tier, results):
    L = []
    overall = "PASS" if (agg_tier[0] not in ("UNKNOWN", "below-Bronze")) else "BELOW-BRONZE"
    L.append(f"# CC6 Tier — Integration {integ} — {overall}")
    L.append("")
    L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC6]**")
    L.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append(f"**Model:** {model or '?'}    **Topology:** {topology or '?'}")
    L.append("")
    L.append("## Verdict")
    L.append("| Axis | Best measured | Tier achieved | Threshold |")
    L.append("|---|---|---|---|")
    L.append(f"| Aggregate t/s | {agg_max if agg_max is not None else '—'} | **{agg_tier[0]}** | {agg_tier[1] if agg_tier[1] else '—'} |")
    L.append(f"| Conc=1 t/s    | {conc1_max if conc1_max is not None else '—'} | **{c1_tier[0]}** | {c1_tier[1] if c1_tier[1] else '—'} |")
    L.append("")
    if results:
        L.append("## Per-row")
        cols = ("conc", "tps_per_req", "tps_aggregate", "ttft_p50_s", "tpot_s", "ITL_p50_s")
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in results:
            row = [str(r.get(k, "")) for k in cols]
            L.append("| " + " | ".join(row) + " |")
    return "\n".join(L) + "\n", overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="CC8 bench JSON path or '-' for stdin")
    ap.add_argument("--integ", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--no-bridge", action="store_true")
    args = ap.parse_args()
    blob = load(args.src)
    results = extract_results(blob)
    model = blob.get("model") if isinstance(blob, dict) else None
    topology = blob.get("topology") if isinstance(blob, dict) else None
    agg_max = best(results, "tps_aggregate")
    conc1_max = conc1_per_req(results) or best(results, "tps_per_req")
    agg_tier = classify(agg_max, AGG_TIERS)
    c1_tier = classify(conc1_max, CONC1_TIERS)

    md, overall = render(args.integ, model, topology, agg_max, conc1_max, agg_tier, c1_tier, results)
    out = args.report or f"{os.path.expanduser('~/AGENT/comms')}/CC6_TIER_{args.integ}.md"
    with open(out, "w") as f:
        f.write(md)
    sys.stdout.write(md)
    print(f"\n[tier_classify] {overall} agg={agg_tier[0]}({agg_max}) conc1={c1_tier[0]}({conc1_max}) → {out}", file=sys.stderr)

    if not args.no_bridge:
        body = f"{overall} integ={args.integ} agg_tier={agg_tier[0]} agg_tps={agg_max} conc1_tier={c1_tier[0]} conc1_tps={conc1_max} report={out}"
        try:
            import subprocess
            subprocess.run([
                "python3", os.path.expanduser("~/AGENT/comms/bridge.py"),
                "post", "--from", "claude-cc6", "--topic", "tier_result", "--body", body,
            ], check=False)
        except Exception:
            pass

    # exit 0 if Bronze+, 1 below
    sys.exit(0 if agg_tier[0] not in ("UNKNOWN", "below-Bronze") else 1)


if __name__ == "__main__":
    main()
