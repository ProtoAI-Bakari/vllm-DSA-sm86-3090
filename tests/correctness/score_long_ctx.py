#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK long-context regression scorer; not a perf bench
# Score long_ctx capture against the embedded needle. PASS when needle-recall
# rate is >= 90% per tier (LEAD W3 stretch — long-context coherence floor).
import argparse, json, os, sys, time
from collections import defaultdict


def load_prompts(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[o["id"]] = o
    return out


def get_text(rec):
    if "text" in rec:
        return rec["text"] or ""
    fp = rec.get("final_payload") or {}
    ch = (fp.get("choices") or [{}])[0]
    return ch.get("text") or (ch.get("message") or {}).get("content") or ""


def score(prompts_path, under_path):
    prompts = load_prompts(prompts_path)
    per_tier_total = defaultdict(int)
    per_tier_correct = defaultdict(int)
    misses = []
    with open(under_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id")
            p = prompts.get(pid)
            if not p:
                continue
            tier = p.get("tier", "?")
            per_tier_total[tier] += 1
            expected = p.get("needle_value_expected", "")
            text = get_text(rec)
            if "error" in rec or not text:
                misses.append({"id": pid, "tier": tier, "reason": rec.get("error", "empty"), "text": text[:120]})
                continue
            # case-insensitive substring match against the lower-cased expected fragment
            if expected.lower() in text.lower():
                per_tier_correct[tier] += 1
            else:
                misses.append({"id": pid, "tier": tier, "expected": expected, "got": text[:200]})
    return per_tier_total, per_tier_correct, misses


def render(integ, totals, corrects, misses):
    L = []
    overall_total = sum(totals.values())
    overall_correct = sum(corrects.values())
    pct = (overall_correct / overall_total * 100) if overall_total else 0.0
    floor = 90.0
    verdict = "PASS" if pct >= floor else "FAIL"
    L.append(f"# CC6 Long-Context Regression — {integ} — {verdict}")
    L.append("")
    L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, agent: CC6]**")
    L.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append(f"**Floor:** needle-recall ≥ {floor:.1f}% overall")
    L.append("")
    L.append(f"Overall recall: **{overall_correct}/{overall_total} = {pct:.2f}%**")
    L.append("")
    L.append("## Per-tier")
    L.append("| Tier | N | Correct | Recall % |")
    L.append("|---|---|---|---|")
    for tier in sorted(totals):
        t = totals[tier]; c = corrects[tier]
        L.append(f"| {tier} | {t} | {c} | {(c/t*100 if t else 0):.2f}% |")
    if misses:
        L.append("")
        L.append("## Sample misses (first 25)")
        L.append("| id | tier | expected | got |")
        L.append("|---|---|---|---|")
        for m in misses[:25]:
            exp = m.get("expected", m.get("reason", "?")).replace("|", "\\|")
            got = (m.get("got") or m.get("text") or "").replace("|", "\\|").replace("\n", " ⏎ ")
            L.append(f"| {m['id']} | {m['tier']} | {exp[:80]} | {got[:80]} |")
    return "\n".join(L) + "\n", verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--under", required=True)
    ap.add_argument("--integ", required=True)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    totals, corrects, misses = score(args.prompts, args.under)
    md, verdict = render(args.integ, totals, corrects, misses)
    out = args.report or f"{os.path.expanduser('~/AGENT/comms')}/CC6_LONGCTX_{args.integ}.md"
    with open(out, "w") as f:
        f.write(md)
    sys.stdout.write(md)
    print(f"\n[score_long_ctx] {verdict} → {out}", file=sys.stderr)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
