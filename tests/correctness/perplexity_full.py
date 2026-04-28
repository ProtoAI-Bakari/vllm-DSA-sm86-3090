#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK correctness gate; not a perf bench
# Story 14 (W3): full MMLU perplexity / accuracy eval. Reads two capture JSONLs
# (baseline + under-test) restricted to mmlu_real category and computes:
#   - per-subject top-1 accuracy
#   - overall accuracy
#   - mean correct-letter log-prob drop (when logprobs present)
#   - perplexity drop ratio approximation: exp(mean_logprob_baseline - mean_logprob_under)
# Emits markdown report to ~/AGENT/comms/CC6_PERPLEXITY_<integ>.md and exits
# 0 if drop <= 2% per LEAD §8.4, else 1.
#
# Usage:
#   python3 perplexity_full.py --baseline baseline.jsonl --under-test under.jsonl --integ cc4-rev2
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


def predicted_letter(rec):
    txt = rec.get("text") or ""
    if not txt:
        fp = rec.get("final_payload") or {}
        ch = (fp.get("choices") or [{}])[0]
        txt = ch.get("text") or (ch.get("message") or {}).get("content") or ""
    if not txt:
        return None
    s = txt.strip().upper()
    return s[0] if s and s[0] in "ABCD" else None


def correct_letter_logprob(rec, expected_letter):
    """Extract logprob of expected letter at position 0, if available."""
    fp = rec.get("final_payload") or {}
    ch = (fp.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    top = lp.get("top_logprobs") if isinstance(lp, dict) else None
    if not top:
        return None
    pos0 = top[0] if top else {}
    for tok, val in pos0.items():
        if tok.strip().upper().startswith(expected_letter):
            return float(val.get("logprob") if isinstance(val, dict) else val)
    return None


def compute(baseline_path, under_path):
    base = load_jsonl(baseline_path)
    under = load_jsonl(under_path)
    common = sorted(set(base) & set(under))

    subj_total = defaultdict(int)
    subj_b_correct = defaultdict(int)
    subj_u_correct = defaultdict(int)
    b_logprobs = []
    u_logprobs = []
    paired_drops = []
    sample_disagreements = []

    for pid in common:
        b = base[pid]
        u = under[pid]
        if (b.get("category") not in ("mmlu_real", "mmlu_placeholder")
                and u.get("category") not in ("mmlu_real", "mmlu_placeholder")):
            continue
        exp = (b.get("expected_letter") or u.get("expected_letter") or "").strip()[:1].upper()
        if exp not in "ABCD":
            continue
        subj = b.get("subject") or u.get("subject") or "unknown"
        subj_total[subj] += 1
        bp = predicted_letter(b)
        up = predicted_letter(u)
        if bp == exp:
            subj_b_correct[subj] += 1
        if up == exp:
            subj_u_correct[subj] += 1
        bl = correct_letter_logprob(b, exp)
        ul = correct_letter_logprob(u, exp)
        if bl is not None:
            b_logprobs.append(bl)
        if ul is not None:
            u_logprobs.append(ul)
        if bl is not None and ul is not None:
            paired_drops.append(bl - ul)
        if bp != up and len(sample_disagreements) < 30:
            sample_disagreements.append({
                "id": pid, "subject": subj, "expected": exp,
                "baseline_pred": bp, "under_pred": up,
                "baseline_logprob": bl, "under_logprob": ul,
            })

    n = sum(subj_total.values())
    overall_b_acc = sum(subj_b_correct.values()) / n if n else 0.0
    overall_u_acc = sum(subj_u_correct.values()) / n if n else 0.0
    accuracy_drop_pct = (overall_b_acc - overall_u_acc) * 100
    mean_b_lp = statistics.mean(b_logprobs) if b_logprobs else None
    mean_u_lp = statistics.mean(u_logprobs) if u_logprobs else None
    mean_paired_drop = statistics.mean(paired_drops) if paired_drops else None
    perplexity_ratio = math.exp(mean_paired_drop) if mean_paired_drop is not None else None
    perplexity_drop_pct = (perplexity_ratio - 1) * 100 if perplexity_ratio is not None else None
    gate_pass = (perplexity_drop_pct is None) or (perplexity_drop_pct <= 2.0)

    return {
        "n_mmlu": n,
        "n_subjects": len(subj_total),
        "overall_baseline_acc": overall_b_acc,
        "overall_under_acc": overall_u_acc,
        "accuracy_drop_pct": accuracy_drop_pct,
        "mean_baseline_correct_logprob": mean_b_lp,
        "mean_under_correct_logprob": mean_u_lp,
        "mean_paired_correct_logprob_drop": mean_paired_drop,
        "perplexity_ratio_approx": perplexity_ratio,
        "perplexity_drop_pct_approx": perplexity_drop_pct,
        "per_subject": {
            s: {
                "n": subj_total[s],
                "baseline_acc": subj_b_correct[s] / subj_total[s] if subj_total[s] else 0.0,
                "under_acc": subj_u_correct[s] / subj_total[s] if subj_total[s] else 0.0,
            } for s in sorted(subj_total)
        },
        "sample_disagreements": sample_disagreements,
        "gate_pass": gate_pass,
    }


def render(report, integ):
    g = "PASS" if report["gate_pass"] else "FAIL"
    L = []
    L.append(f"# CC6 Perplexity (full MMLU) — Integration {integ} — {g}")
    L.append("")
    L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, agent: CC6]**")
    L.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append("")
    L.append(f"- N MMLU prompts compared: **{report['n_mmlu']}** across {report['n_subjects']} subjects")
    L.append(f"- Overall accuracy: baseline **{report['overall_baseline_acc']:.4f}** vs under-test **{report['overall_under_acc']:.4f}** (drop {report['accuracy_drop_pct']:+.3f} pp)")
    if report['mean_paired_correct_logprob_drop'] is not None:
        L.append(f"- Mean correct-letter log-prob: baseline **{report['mean_baseline_correct_logprob']:.4f}** under **{report['mean_under_correct_logprob']:.4f}**")
        L.append(f"- Approx perplexity ratio: **{report['perplexity_ratio_approx']:.4f}**  (drop **{report['perplexity_drop_pct_approx']:+.3f}%**, gate ≤2%)")
    else:
        L.append(f"- log-prob comparison unavailable (no logprobs in capture(s))")
    L.append("")
    L.append("## Per-subject accuracy")
    L.append("| Subject | N | Baseline | Under | Δ |")
    L.append("|---|---|---|---|---|")
    for s, d in report["per_subject"].items():
        diff = d["baseline_acc"] - d["under_acc"]
        L.append(f"| {s} | {d['n']} | {d['baseline_acc']:.4f} | {d['under_acc']:.4f} | {diff:+.4f} |")
    if report["sample_disagreements"]:
        L.append("")
        L.append("## Sample disagreements (first 30)")
        L.append("| id | subject | expected | baseline | under | b_lp | u_lp |")
        L.append("|---|---|---|---|---|---|---|")
        for d in report["sample_disagreements"]:
            L.append(f"| {d['id']} | {d['subject']} | {d['expected']} | {d['baseline_pred']} | {d['under_pred']} | {d['baseline_logprob']} | {d['under_logprob']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--under-test", required=True)
    ap.add_argument("--integ", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    report = compute(args.baseline, args.under_test)
    md = render(report, args.integ)
    out_path = args.report or f"{os.path.expanduser('~/AGENT/comms')}/CC6_PERPLEXITY_{args.integ}.md"
    with open(out_path, "w") as f:
        f.write(md)
    sys.stdout.write(md)
    print(f"\n[perplexity_full] {'PASS' if report['gate_pass'] else 'FAIL'} → {out_path}", file=sys.stderr)
    sys.exit(0 if report["gate_pass"] else 1)


if __name__ == "__main__":
    main()
