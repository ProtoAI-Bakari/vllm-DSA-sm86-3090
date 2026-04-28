#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK capture aggregator; not a perf bench
# Merge N per-rank capture JSONLs (eg from PP=2+TP=8 where each rank dumps its
# slice) into a single canonical JSONL the gate can grade. Resolves duplicate
# ids by preferring rank 0 deterministically; flags genuine disagreements.
#
# Run:
#   python3 merge_captures.py --in rank0.jsonl rank1.jsonl rank2.jsonl --out merged.jsonl
#   python3 merge_captures.py --in 'captures_*.jsonl' --out merged.jsonl --integ cc4-rev2
import argparse, glob, json, sys
from collections import defaultdict


def expand(paths):
    out = []
    for p in paths:
        if any(c in p for c in "*?["):
            out.extend(sorted(glob.glob(p)))
        else:
            out.append(p)
    return out


def load(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def text_of(rec):
    if "text" in rec:
        return rec["text"] or ""
    fp = rec.get("final_payload") or {}
    ch = (fp.get("choices") or [{}])[0]
    return ch.get("text") or (ch.get("message") or {}).get("content") or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--integ", default=None)
    ap.add_argument("--strict-disagree", action="store_true",
                    help="exit 1 on any text mismatch across ranks for the same id")
    args = ap.parse_args()
    paths = expand(args.src)
    if not paths:
        print(f"[merge_captures] no input files matched {args.src}", file=sys.stderr)
        sys.exit(2)

    # rank order = file order on cmdline (rank0 first)
    by_id = {}
    disagreements = []
    n_total = 0
    n_kept = 0
    n_dup = 0
    by_rank_first = defaultdict(int)

    for rank_idx, path in enumerate(paths):
        recs = load(path)
        n_total += len(recs)
        for r in recs:
            pid = r.get("id")
            if not pid:
                continue
            if pid not in by_id:
                r2 = dict(r)
                r2["__source_rank__"] = rank_idx
                r2["__source_path__"] = path
                if args.integ:
                    r2["integ"] = args.integ
                by_id[pid] = r2
                by_rank_first[rank_idx] += 1
                n_kept += 1
            else:
                n_dup += 1
                # check for disagreement
                t_old = text_of(by_id[pid])
                t_new = text_of(r)
                if t_old != t_new and "error" not in by_id[pid] and "error" not in r:
                    disagreements.append({
                        "id": pid,
                        "rank_kept": by_id[pid].get("__source_rank__"),
                        "rank_dup": rank_idx,
                        "text_kept": t_old[:120],
                        "text_dup": t_new[:120],
                    })

    with open(args.out, "w") as f:
        for pid in sorted(by_id):
            f.write(json.dumps(by_id[pid], ensure_ascii=False) + "\n")

    print(f"[merge_captures] inputs={len(paths)} total_recs={n_total} unique_ids={n_kept} duplicates={n_dup} disagreements={len(disagreements)}", file=sys.stderr)
    print(f"[merge_captures] per-rank kept-first: {dict(by_rank_first)}", file=sys.stderr)
    print(f"[merge_captures] wrote {args.out}", file=sys.stderr)
    if disagreements:
        print(f"[merge_captures] first 5 disagreements:", file=sys.stderr)
        for d in disagreements[:5]:
            print(f"  id={d['id']} rank{d['rank_kept']}={d['text_kept']!r} rank{d['rank_dup']}={d['text_dup']!r}", file=sys.stderr)
        if args.strict_disagree:
            sys.exit(1)


if __name__ == "__main__":
    main()
