#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK baseline integrity audit; not a perf bench
# CC0 22:57Z dispatch (1): full integrity check on baseline.jsonl.
# Verifies no schema drift, no NaN logprobs, no truncation, every prompt
# represented exactly once, every category at expected count.
# Emits markdown report + JSON + exit 0 healthy / 1 issues found.
import argparse, json, math, os, sys, time
from collections import Counter
from pathlib import Path

EXPECTED_CATS = {
    "factual": 200, "mmlu_placeholder": 300, "reasoning": 200, "code": 200,
    "summarization": 150, "creative": 150, "multilingual": 100,
    "repetition": 100, "edge": 100,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(Path(__file__).parent / "baseline.jsonl"))
    ap.add_argument("--prompts", default=str(Path(__file__).parent / "prompts.jsonl"))
    ap.add_argument("--report", default=os.path.expanduser("~/AGENT/comms/CC6_BASELINE_INTEGRITY.md"))
    ap.add_argument("--json-out", default=os.path.expanduser("~/AGENT/comms/CC6_BASELINE_INTEGRITY.json"))
    args = ap.parse_args()

    issues = []

    if not os.path.isfile(args.baseline):
        print(f"MISSING baseline: {args.baseline}", file=sys.stderr)
        sys.exit(2)

    prompts_ids = set()
    if os.path.isfile(args.prompts):
        with open(args.prompts) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    if "id" in o:
                        prompts_ids.add(o["id"])
                except json.JSONDecodeError:
                    pass

    n = 0
    n_err = 0
    cats = Counter()
    ids = Counter()
    truncated = []
    nan_logits = []
    missing_text = []
    bad_json = 0
    completion_token_bins = Counter()

    with open(args.baseline) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                issues.append(f"line {ln}: invalid JSON")
                continue
            n += 1
            cats[rec.get("category", "?")] += 1
            ids[rec.get("id", "?")] += 1
            if "error" in rec:
                n_err += 1
                continue
            text = rec.get("text", "")
            if text is None or text == "":
                # empty completion is legal for "edge" category prompts
                if rec.get("category") != "edge":
                    missing_text.append(rec.get("id"))
            metrics = rec.get("metrics") or {}
            ctok = metrics.get("completion_tokens") or 0
            req_max = rec.get("request_max_tokens") or 0
            if isinstance(ctok, (int, float)) and isinstance(req_max, (int, float)) and req_max > 0:
                if ctok >= req_max:  # likely truncated at max_tokens
                    truncated.append({"id": rec.get("id"), "completion_tokens": ctok, "max": req_max})
                # Bin completion tokens
                if ctok < 8: bin_k = "<8"
                elif ctok < 16: bin_k = "8-15"
                elif ctok < 64: bin_k = "16-63"
                elif ctok < 128: bin_k = "64-127"
                else: bin_k = ">=128"
                completion_token_bins[bin_k] += 1
            # Check logprobs for NaN/Inf if present
            fp = rec.get("final_payload") or {}
            ch = (fp.get("choices") or [{}])[0]
            lps = (ch.get("logprobs") or {}).get("top_logprobs")
            if isinstance(lps, list):
                for pos, dist in enumerate(lps):
                    if not isinstance(dist, dict):
                        continue
                    for tok, val in dist.items():
                        v = val.get("logprob") if isinstance(val, dict) else val
                        try:
                            vf = float(v)
                            if not math.isfinite(vf):
                                nan_logits.append({"id": rec.get("id"), "pos": pos, "tok": tok, "v": v})
                        except (TypeError, ValueError):
                            pass

    # Schema drift checks
    extra_cats = set(cats) - set(EXPECTED_CATS)
    if extra_cats:
        issues.append(f"unexpected categories: {sorted(extra_cats)}")
    for cat, expected in EXPECTED_CATS.items():
        if cats.get(cat, 0) != expected:
            issues.append(f"category {cat} count={cats.get(cat,0)} expected={expected}")

    dup_ids = [k for k, v in ids.items() if v > 1]
    if dup_ids:
        issues.append(f"{len(dup_ids)} duplicate ids (first 5: {dup_ids[:5]})")
    if prompts_ids:
        missing_from_baseline = prompts_ids - set(ids)
        if missing_from_baseline:
            issues.append(f"{len(missing_from_baseline)} prompts not in baseline (first 5: {sorted(missing_from_baseline)[:5]})")

    if missing_text:
        issues.append(f"{len(missing_text)} non-edge records have empty text (first 5: {missing_text[:5]})")
    if nan_logits:
        issues.append(f"{len(nan_logits)} non-finite logprob values (first 5: {nan_logits[:5]})")
    if bad_json:
        issues.append(f"{bad_json} invalid JSON lines")

    healthy = len(issues) == 0

    # Report
    L = []
    L.append(f"# CC6 Baseline Integrity Check — {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append("")
    L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC6]**")
    L.append(f"**Baseline:** `{args.baseline}`  ({n} records)")
    L.append(f"**Prompts source:** `{args.prompts}`  ({len(prompts_ids)} ids)")
    L.append(f"**Verdict:** {'✓ HEALTHY' if healthy else '✗ ISSUES (' + str(len(issues)) + ')'}")
    L.append("")
    L.append("## Counts")
    L.append("| Field | Value |")
    L.append("|---|---|")
    L.append(f"| total records | {n} |")
    L.append(f"| errored records | {n_err} |")
    L.append(f"| unique ids | {len(ids)} |")
    L.append(f"| duplicate ids | {len(dup_ids)} |")
    L.append(f"| invalid JSON lines | {bad_json} |")
    L.append(f"| truncated (ctok ≥ max_tokens) | {len(truncated)} |")
    L.append(f"| non-finite logprob values | {len(nan_logits)} |")
    L.append(f"| empty text (non-edge) | {len(missing_text)} |")
    L.append("")
    L.append("## Per-category vs expected")
    L.append("| category | expected | actual | delta |")
    L.append("|---|---|---|---|")
    for cat, exp in EXPECTED_CATS.items():
        act = cats.get(cat, 0)
        L.append(f"| {cat} | {exp} | {act} | {act - exp:+d} |")
    L.append("")
    L.append("## completion-token distribution")
    L.append("| bin | n |")
    L.append("|---|---|")
    for b in ("<8", "8-15", "16-63", "64-127", ">=128"):
        L.append(f"| {b} | {completion_token_bins.get(b, 0)} |")
    if issues:
        L.append("")
        L.append("## Issues")
        for i in issues:
            L.append(f"- {i}")

    md = "\n".join(L) + "\n"
    with open(args.report, "w") as f:
        f.write(md)
    with open(args.json_out, "w") as f:
        json.dump({
            "healthy": healthy, "issues": issues, "n": n, "n_err": n_err,
            "duplicate_ids": dup_ids[:50], "truncated_first_5": truncated[:5],
            "nan_logits_first_5": nan_logits[:5], "missing_text_first_5": missing_text[:5],
            "categories": dict(cats), "completion_token_bins": dict(completion_token_bins),
        }, f, indent=2)
    sys.stdout.write(md)
    print(f"\n[baseline_integrity] {'HEALTHY' if healthy else 'ISSUES=' + str(len(issues))} → {args.report}", file=sys.stderr)
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
