#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK rev-comparison tool; not a perf bench
# Compare baseline + N under-test JSONLs side-by-side for the same prompt id.
# Useful when CC0 evaluates a sequence of CC4 revs (rev1/rev2/rev3) — instead
# of N separate verify_gate reports, see one matrix showing which ids each rev
# agreed/disagreed with baseline on.
#
# Run:
#   python3 compare_revs.py --baseline base.jsonl --revs rev1.jsonl rev2.jsonl rev3.jsonl
#   python3 compare_revs.py ... --top 30   # show top 30 mismatched ids
import argparse, json, sys
from collections import defaultdict


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in rec:
                out[rec["id"]] = rec
    return out


def text_of(rec):
    if rec is None:
        return None
    if "error" in rec:
        return f"<error: {rec['error'][:60]}>"
    if "text" in rec:
        return rec["text"] or ""
    fp = rec.get("final_payload") or {}
    ch = (fp.get("choices") or [{}])[0]
    return ch.get("text") or (ch.get("message") or {}).get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--revs", nargs="+", required=True, help="under-test JSONL paths in order rev1 rev2 ...")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--md-out", default=None)
    args = ap.parse_args()

    base = load(args.baseline)
    revs = [(p, load(p)) for p in args.revs]

    common = sorted(set(base) & set.intersection(*(set(r) for _, r in revs)) if revs else set(base))
    n_common = len(common)
    agree_per_rev = [0] * len(revs)
    error_per_rev = [0] * len(revs)
    only_rev_disagrees = [0] * len(revs)
    rows = []
    for pid in common:
        b = base[pid]
        bt = text_of(b)
        rev_texts = [text_of(r.get(pid)) for _, r in revs]
        agreements = [bt == rt for rt in rev_texts]
        for i, ag in enumerate(agreements):
            if ag:
                agree_per_rev[i] += 1
            if "<error" in (rev_texts[i] or ""):
                error_per_rev[i] += 1
        if not all(agreements):
            disagreers = sum(1 for a in agreements if not a)
            if disagreers == 1:
                idx = agreements.index(False)
                only_rev_disagrees[idx] += 1
            rows.append({"id": pid, "baseline": bt, "revs": rev_texts, "agree": agreements})

    L = []
    L.append("# CC6 rev comparison")
    L.append("")
    L.append(f"Baseline: `{args.baseline}` ({len(base)} records)")
    for i, (p, r) in enumerate(revs):
        agr = agree_per_rev[i] / n_common if n_common else 0
        L.append(f"- rev{i+1}: `{p}` — agreement={agree_per_rev[i]}/{n_common} ({agr*100:.2f}%) errors={error_per_rev[i]} only-this-rev-disagrees={only_rev_disagrees[i]}")
    L.append("")
    if rows:
        L.append(f"## Top {min(args.top, len(rows))} disagreements")
        L.append("| id | baseline | " + " | ".join(f"rev{i+1}" for i in range(len(revs))) + " |")
        L.append("|---|---|" + "|".join(["---"] * len(revs)) + "|")
        for r in rows[:args.top]:
            cells = [r["id"], (r["baseline"] or "")[:60].replace("|", "\\|").replace("\n", " ⏎ ")]
            for rt, ag in zip(r["revs"], r["agree"]):
                tag = "" if ag else "✗ "
                cells.append(tag + (rt or "")[:60].replace("|", "\\|").replace("\n", " ⏎ "))
            L.append("| " + " | ".join(cells) + " |")
    md = "\n".join(L) + "\n"
    sys.stdout.write(md)
    if args.md_out:
        with open(args.md_out, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
