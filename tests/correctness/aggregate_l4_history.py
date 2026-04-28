#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK history aggregator; not a perf bench
# Walks ~/AGENT/comms/ for CC6_GATE_RESULTS_*.md, CC6_PERPLEXITY_*.md,
# CC6_LONGCTX_*.md, CC6_TIER_*.md, CC6_GRILL33_*.md and aggregates the
# per-integration verdicts into a single CSV + markdown trend report.
#
# Lets CC0 see whether the gate is converging (PASS streaks) or thrashing
# (FAIL→PASS→FAIL bouncing) across CC3/CC4/CC5 ships over time.
#
# Run:
#   python3 aggregate_l4_history.py
#   python3 aggregate_l4_history.py --csv-out ~/AGENT/comms/CC6_HISTORY.csv
import argparse, csv, glob, os, re, sys, time
from datetime import datetime
from pathlib import Path

COMMS = Path(os.path.expanduser("~/AGENT/comms"))


PATTERNS = {
    "gate":       "CC6_GATE_RESULTS_*.md",
    "perplexity": "CC6_PERPLEXITY_*.md",
    "longctx":    "CC6_LONGCTX_*.md",
    "tier":       "CC6_TIER_*.md",
    "grill33":    "CC6_GRILL33_*.md",
}

VERDICT_RE = re.compile(r"#\s*CC6\s*[A-Za-z _0-9-]*—\s*[A-Za-z 0-9-]*—\s*(✓?\s*PASS|✗?\s*FAIL|BELOW-BRONZE)", re.I)
INTEG_RE = re.compile(r"^.*?_(.+)\.md$")


def parse_md(path, kind):
    text = path.read_text(errors="replace")
    m = VERDICT_RE.search(text)
    verdict = m.group(1).strip().replace("✓ ", "").replace("✗ ", "") if m else "UNKNOWN"
    fname = path.name
    integ = re.sub(r"^CC6_(GATE_RESULTS|PERPLEXITY|LONGCTX|TIER|GRILL33)_", "", fname).rsplit(".md", 1)[0]
    mtime = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    # Capture top-1 / cosine / perplexity numbers from the markdown summary table when present
    metrics = {}
    for line in text.splitlines():
        s = line.strip()
        for k_label, key in (
            ("top-1 text match", "top1_pct"),
            ("mean logit cosine", "cosine"),
            ("MMLU perplexity drop", "ppl_drop_pct"),
            ("mean word Jaccard", "jaccard"),
            ("Aggregate t/s", "agg_tps"),
            ("Conc=1 t/s", "conc1_tps"),
        ):
            if s.startswith(f"| {k_label}") and "|" in s:
                cells = [c.strip() for c in s.strip("|").split("|")]
                if len(cells) >= 2:
                    metrics[key] = cells[1]
    return {"kind": kind, "integ": integ, "verdict": verdict, "ts": mtime, "path": str(path), **metrics}


def collect():
    rows = []
    for kind, pat in PATTERNS.items():
        for f in COMMS.glob(pat):
            try:
                rows.append(parse_md(f, kind))
            except Exception as e:
                print(f"  WARN: {f}: {e}", file=sys.stderr)
    rows.sort(key=lambda r: r["ts"])
    return rows


def render_md(rows):
    L = []
    L.append(f"# CC6 L4 History — {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    L.append("")
    L.append(f"**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [agent: CC6]**")
    L.append(f"**Total verdict files:** {len(rows)}")
    L.append("")
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)
    L.append("## Summary by report kind")
    L.append("| Kind | N | PASS | FAIL | UNKNOWN |")
    L.append("|---|---|---|---|---|")
    for kind in sorted(by_kind):
        rs = by_kind[kind]
        p = sum(1 for r in rs if "PASS" in r["verdict"])
        f = sum(1 for r in rs if "FAIL" in r["verdict"] or "BELOW" in r["verdict"])
        u = len(rs) - p - f
        L.append(f"| {kind} | {len(rs)} | {p} | {f} | {u} |")
    L.append("")
    L.append("## Recent verdicts (last 30, newest first)")
    L.append("| ts | kind | integ | verdict | top1 | cosine | ppl% | jaccard | agg_tps | conc1_tps |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in reversed(rows[-30:]):
        L.append(f"| {r['ts']} | {r['kind']} | {r['integ']} | **{r['verdict']}** | "
                 f"{r.get('top1_pct','')} | {r.get('cosine','')} | {r.get('ppl_drop_pct','')} | "
                 f"{r.get('jaccard','')} | {r.get('agg_tps','')} | {r.get('conc1_tps','')} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-out", default=str(COMMS / "CC6_HISTORY.csv"))
    ap.add_argument("--md-out", default=str(COMMS / "CC6_HISTORY.md"))
    args = ap.parse_args()
    rows = collect()
    fields = ["ts", "kind", "integ", "verdict", "top1_pct", "cosine",
              "ppl_drop_pct", "jaccard", "agg_tps", "conc1_tps", "path"]
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = render_md(rows)
    with open(args.md_out, "w") as f:
        f.write(md)
    sys.stdout.write(md)
    print(f"\n[aggregate_l4_history] rows={len(rows)} → {args.md_out} + {args.csv_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
