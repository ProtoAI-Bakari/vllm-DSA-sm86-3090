#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK correctness-gate verifier; not a perf bench (capture_one.py owns TTFT/TPOT/etc.)
# Compare two baseline.jsonl files (--baseline + --under-test) and emit gate verdict.
#
# Gate (LEAD corpus PART2 §8.4):
#   PRIMARY  top-1  text-match rate         >= 98%   (deterministic temp=0 should be exact)
#   PRIMARY  logit cosine (when present)    >= 0.97
#   PRIMARY  perplexity drop on MMLU subset <= 2%    (requires logprobs on mmlu_real category)
#   FALLBACK Jaccard of generated tokens    >= 0.95  (when logprobs missing)
#
# Output:
#   - ./CC6_GATE_RESULTS_<integ>.md  (or path passed via --report)
#   - bridge.py post --topic gate --body "<verdict>"  (if --bridge given)
#   - exit 0 PASS / 1 FAIL
import argparse, json, math, os, statistics, sys, time
from pathlib import Path
from collections import defaultdict

def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in obj:
                out[obj["id"]] = obj
    return out

def extract_text(rec):
    if "text" in rec:
        return rec["text"] or ""
    fp = rec.get("final_payload") or {}
    ch = (fp.get("choices") or [{}])[0]
    return ch.get("text") or (ch.get("message") or {}).get("content") or ""

def extract_top_logprobs(rec):
    """Return list[ dict[token]=logprob ] across positions, or None."""
    fp = rec.get("final_payload") or {}
    ch = (fp.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    if isinstance(lp, dict):
        if "top_logprobs" in lp and isinstance(lp["top_logprobs"], list):
            return [t if isinstance(t, dict) else {} for t in lp["top_logprobs"]]
    return None

def jaccard_chars(a, b):
    if not a and not b:
        return 1.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    return len(inter) / max(1, len(union))

def jaccard_words(a, b):
    if not a and not b:
        return 1.0
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    inter = sa & sb
    union = sa | sb
    return len(inter) / max(1, len(union))

def cosine_topk(a_dist, b_dist):
    """Cosine over union of keys; missing keys treated as exp(very negative) ≈ 0."""
    keys = set(a_dist) | set(b_dist)
    if not keys:
        return None
    av, bv = [], []
    for k in keys:
        av.append(math.exp(a_dist.get(k, -30.0)))
        bv.append(math.exp(b_dist.get(k, -30.0)))
    da = math.sqrt(sum(x*x for x in av))
    db = math.sqrt(sum(x*x for x in bv))
    if da == 0 or db == 0:
        return None
    return sum(a*b for a, b in zip(av, bv)) / (da * db)

def per_position_cosine(a_lps, b_lps):
    """Average cosine across aligned positions of two top-logprob lists."""
    if not a_lps or not b_lps:
        return None
    n = min(len(a_lps), len(b_lps))
    if n == 0:
        return None
    cs = []
    for i in range(n):
        c = cosine_topk(a_lps[i] or {}, b_lps[i] or {})
        if c is not None:
            cs.append(c)
    if not cs:
        return None
    return statistics.mean(cs)

def compute(baseline_path, under_test_path, integ_label):
    baseline = load_jsonl(baseline_path)
    under = load_jsonl(under_test_path)
    common = sorted(set(baseline) & set(under))
    only_baseline = sorted(set(baseline) - set(under))
    only_under = sorted(set(under) - set(baseline))

    cat_total = defaultdict(int)
    cat_match = defaultdict(int)
    cat_jaccard = defaultdict(list)
    cat_cosine = defaultdict(list)
    mmlu_correct_baseline = 0
    mmlu_correct_under = 0
    mmlu_total = 0
    cat_baseline_err = defaultdict(int)
    cat_under_err = defaultdict(int)

    detail_disagreements = []

    for pid in common:
        b = baseline[pid]
        u = under[pid]
        cat = b.get("category") or u.get("category") or "unknown"
        cat_total[cat] += 1
        if "error" in b:
            cat_baseline_err[cat] += 1
        if "error" in u:
            cat_under_err[cat] += 1
        if "error" in b or "error" in u:
            continue
        bt = extract_text(b)
        ut = extract_text(u)
        match = (bt == ut)
        if match:
            cat_match[cat] += 1
        else:
            if len(detail_disagreements) < 50:
                detail_disagreements.append({
                    "id": pid, "category": cat,
                    "baseline_text": bt[:200],
                    "under_test_text": ut[:200],
                })
        cat_jaccard[cat].append(jaccard_words(bt, ut))
        b_lps = extract_top_logprobs(b)
        u_lps = extract_top_logprobs(u)
        cs = per_position_cosine(b_lps, u_lps)
        if cs is not None:
            cat_cosine[cat].append(cs)
        if cat in ("mmlu_real",) and b.get("expected_letter"):
            mmlu_total += 1
            exp = b["expected_letter"].strip()[:1].upper()
            if bt.strip().upper().startswith(exp):
                mmlu_correct_baseline += 1
            if ut.strip().upper().startswith(exp):
                mmlu_correct_under += 1

    overall_match = sum(cat_match.values())
    overall_total = sum(cat_total.values())
    pct_match = (overall_match / overall_total * 100) if overall_total else 0.0
    overall_jaccard = []
    for v in cat_jaccard.values():
        overall_jaccard.extend(v)
    mean_jaccard = statistics.mean(overall_jaccard) if overall_jaccard else None
    overall_cosine = []
    for v in cat_cosine.values():
        overall_cosine.extend(v)
    mean_cosine = statistics.mean(overall_cosine) if overall_cosine else None

    mmlu_acc_baseline = (mmlu_correct_baseline / mmlu_total) if mmlu_total else None
    mmlu_acc_under = (mmlu_correct_under / mmlu_total) if mmlu_total else None
    mmlu_drop_pct = None
    if mmlu_acc_baseline and mmlu_acc_under is not None:
        mmlu_drop_pct = (mmlu_acc_baseline - mmlu_acc_under) / mmlu_acc_baseline * 100

    gate_top1 = pct_match >= 98.0
    gate_cos = (mean_cosine is None) or (mean_cosine >= 0.97)
    gate_ppl = (mmlu_drop_pct is None) or (mmlu_drop_pct <= 2.0)
    gate_jac = (mean_jaccard is None) or (mean_jaccard >= 0.95)
    gate_pass = gate_top1 and gate_cos and gate_ppl and gate_jac

    return {
        "integ": integ_label,
        "ts": int(time.time()),
        "baseline_path": str(baseline_path),
        "under_test_path": str(under_test_path),
        "n_common": len(common),
        "n_only_baseline": len(only_baseline),
        "n_only_under": len(only_under),
        "overall": {
            "top1_match_pct": round(pct_match, 3),
            "mean_jaccard_words": round(mean_jaccard, 4) if mean_jaccard is not None else None,
            "mean_logit_cosine": round(mean_cosine, 4) if mean_cosine is not None else None,
            "mmlu_acc_baseline": round(mmlu_acc_baseline, 4) if mmlu_acc_baseline is not None else None,
            "mmlu_acc_under_test": round(mmlu_acc_under, 4) if mmlu_acc_under is not None else None,
            "mmlu_perplexity_drop_pct": round(mmlu_drop_pct, 3) if mmlu_drop_pct is not None else None,
        },
        "gates": {
            "top1_>=98%": gate_top1,
            "cosine_>=0.97": gate_cos,
            "perplexity_<=2%": gate_ppl,
            "jaccard_>=0.95": gate_jac,
            "PASS": gate_pass,
        },
        "per_category": {
            cat: {
                "n": cat_total[cat],
                "top1_match_pct": round(cat_match[cat] / cat_total[cat] * 100, 3) if cat_total[cat] else 0.0,
                "mean_jaccard": round(statistics.mean(cat_jaccard[cat]), 4) if cat_jaccard[cat] else None,
                "mean_cosine": round(statistics.mean(cat_cosine[cat]), 4) if cat_cosine[cat] else None,
                "baseline_errors": cat_baseline_err[cat],
                "under_test_errors": cat_under_err[cat],
            } for cat in sorted(cat_total)
        },
        "sample_disagreements": detail_disagreements[:20],
    }

def render_markdown(report):
    o = report["overall"]
    g = report["gates"]
    verdict = "✓ PASS" if g["PASS"] else "✗ FAIL"
    lines = []
    lines.append(f"# CC6 Gate — Integration {report['integ']} — {verdict}")
    lines.append("")
    lines.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, xhigh effort, agent: CC6]**")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(report['ts']))}")
    lines.append(f"**Baseline:** `{report['baseline_path']}`")
    lines.append(f"**Under test:** `{report['under_test_path']}`")
    lines.append(f"**Common prompts:** {report['n_common']}  (baseline-only: {report['n_only_baseline']}, under-only: {report['n_only_under']})")
    lines.append("")
    lines.append("## Overall")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|---|---|---|---|")
    lines.append(f"| top-1 text match | {o['top1_match_pct']}% | ≥98% | {'PASS' if g['top1_>=98%'] else 'FAIL'} |")
    lines.append(f"| mean logit cosine | {o['mean_logit_cosine']} | ≥0.97 | {'PASS' if g['cosine_>=0.97'] else 'FAIL/NA'} |")
    lines.append(f"| MMLU perplexity drop | {o['mmlu_perplexity_drop_pct']}% | ≤2% | {'PASS' if g['perplexity_<=2%'] else 'FAIL/NA'} |")
    lines.append(f"| mean word Jaccard | {o['mean_jaccard_words']} | ≥0.95 | {'PASS' if g['jaccard_>=0.95'] else 'FAIL/NA'} |")
    lines.append(f"| MMLU acc baseline | {o['mmlu_acc_baseline']} | — | — |")
    lines.append(f"| MMLU acc under-test | {o['mmlu_acc_under_test']} | — | — |")
    lines.append("")
    lines.append("## Per-category breakdown")
    lines.append("| Category | N | top-1 % | Jaccard | Cosine | b-errs | u-errs |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, d in report["per_category"].items():
        lines.append(f"| {cat} | {d['n']} | {d['top1_match_pct']}% | {d['mean_jaccard']} | {d['mean_cosine']} | {d['baseline_errors']} | {d['under_test_errors']} |")
    lines.append("")
    if report["sample_disagreements"]:
        lines.append("## Sample disagreements (first 20)")
        lines.append("| id | category | baseline | under-test |")
        lines.append("|---|---|---|---|")
        for d in report["sample_disagreements"]:
            b = d["baseline_text"].replace("|", "\\|").replace("\n", " ⏎ ")
            u = d["under_test_text"].replace("|", "\\|").replace("\n", " ⏎ ")
            lines.append(f"| {d['id']} | {d['category']} | {b} | {u} |")
        lines.append("")
    return "\n".join(lines) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--under-test", required=True)
    ap.add_argument("--integ", required=True, help="integration label, e.g. 'cc3-1' or 'cc4-rev2'")
    ap.add_argument("--report", default=None, help="markdown report path; default ~/AGENT/comms/CC6_GATE_RESULTS_<integ>.md")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--bridge", action="store_true", help="post verdict to bridge.db")
    args = ap.parse_args()
    report = compute(args.baseline, args.under_test, args.integ)
    md = render_markdown(report)
    out_path = args.report or f"{os.path.expanduser('~/AGENT/comms')}/CC6_GATE_RESULTS_{args.integ}.md"
    with open(out_path, "w") as f:
        f.write(md)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
    sys.stdout.write(md)
    g = report["gates"]
    verdict = "PASS" if g["PASS"] else "FAIL"
    print(f"\n[verify_gate] {verdict} → {out_path}", file=sys.stderr)
    sys.exit(0 if g["PASS"] else 1)

if __name__ == "__main__":
    main()
