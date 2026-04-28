#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK pre-DSV4 baseline-reference freezer; not a perf bench
# CC0 22:57Z dispatch (2): freeze a reference manifest of the GLM-5.1 baseline
# so we can A/B vs DSV4-Flash-FP8 once it serves. Since the MLX baseline lacks
# logprobs (MLX-LM constraint), perplexity is unavailable — instead we freeze
# text-level statistics (token counts, length distributions, per-category
# answer fingerprints via short hash) plus per-id text. Used by:
#    perplexity_full.py --baseline ... (when DSV4 has logprobs)
#    cycle6_smoke grading (text-level)
#
# Run: python3 freeze_baseline_reference.py [--in baseline.jsonl] [--out reference.json]
import argparse, hashlib, json, os, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path


def sha8(s):
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(Path(__file__).parent / "baseline.jsonl"))
    ap.add_argument("--out", default=str(Path(__file__).parent / "baseline_reference.json"))
    args = ap.parse_args()

    if not os.path.isfile(args.src):
        print(f"MISSING: {args.src}", file=sys.stderr)
        sys.exit(2)

    by_cat = defaultdict(lambda: {"n": 0, "ctok": [], "wall": [], "len_chars": []})
    per_id_fingerprint = {}
    n = 0
    n_err = 0

    with open(args.src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if "error" in rec:
                n_err += 1
                continue
            cat = rec.get("category", "?")
            text = rec.get("text") or ""
            metrics = rec.get("metrics") or {}
            by_cat[cat]["n"] += 1
            ctok = metrics.get("completion_tokens")
            if isinstance(ctok, (int, float)):
                by_cat[cat]["ctok"].append(float(ctok))
            wall = metrics.get("wall_s")
            if isinstance(wall, (int, float)):
                by_cat[cat]["wall"].append(float(wall))
            by_cat[cat]["len_chars"].append(len(text))
            per_id_fingerprint[rec["id"]] = {
                "cat": cat,
                "len": len(text),
                "ctok": ctok,
                "sha8": sha8(text),
                "first40": text[:40],
            }

    cat_summary = {}
    for cat, d in by_cat.items():
        cat_summary[cat] = {
            "n": d["n"],
            "ctok_p50": statistics.median(d["ctok"]) if d["ctok"] else None,
            "ctok_mean": statistics.mean(d["ctok"]) if d["ctok"] else None,
            "ctok_max": max(d["ctok"]) if d["ctok"] else None,
            "wall_p50": statistics.median(d["wall"]) if d["wall"] else None,
            "len_chars_p50": statistics.median(d["len_chars"]) if d["len_chars"] else None,
            "len_chars_p95": sorted(d["len_chars"])[int(0.95 * (len(d["len_chars"]) - 1))] if len(d["len_chars"]) > 1 else None,
        }

    out = {
        "_meta": {
            "source": args.src,
            "n_records": n,
            "n_errors": n_err,
            "category_summary_keys": "n, ctok_{p50,mean,max}, wall_p50, len_chars_{p50,p95}",
            "fingerprint_keys": "cat, len, ctok, sha8 (first 8 hex of SHA-256 over text), first40 (first 40 chars)",
            "purpose": "PRE-DSV4 frozen GLM-5.1 baseline reference for A/B vs DSV4-Flash-FP8",
        },
        "category_summary": cat_summary,
        "per_id_fingerprint": per_id_fingerprint,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[freeze_baseline_reference] froze {n} records ({n_err} errors) → {args.out}", file=sys.stderr)
    print(f"[freeze_baseline_reference] {len(by_cat)} categories: " +
          ", ".join(f"{c}={by_cat[c]['n']}" for c in sorted(by_cat)), file=sys.stderr)


if __name__ == "__main__":
    main()
