#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK gate-input adapter; not a perf bench
# CC0 directive (a): consume v0.20.0-rebased capture format. CC9 may ship
# captures from rebuilt vLLM 0.20.0 which adds metadata fields (engine_version,
# build_sha, kernel_arch, ...) but otherwise matches the 0.19.x capture schema
# verify_gate.py already understands.
#
# Confirms format compatibility, surfaces version metadata in the capture, and
# enriches each record with `__capture_version__` so downstream verify_gate
# reports include build provenance.
#
# Run:
#   python3 v020_capture_adapter.py --in capture.jsonl --out enriched.jsonl
#   python3 v020_capture_adapter.py --inspect capture.jsonl     # just print metadata
import argparse, collections, json, sys


def detect_version(records):
    """Sniff records for engine_version-style fields. Returns dict summary."""
    versions = collections.Counter()
    builds = collections.Counter()
    archs = collections.Counter()
    has_logprobs = 0
    has_metrics = 0
    n = 0
    for rec in records:
        n += 1
        v = (rec.get("engine_version") or
             rec.get("vllm_version") or
             (rec.get("metadata") or {}).get("engine_version") or
             (rec.get("metadata") or {}).get("vllm_version"))
        if v:
            versions[str(v)] += 1
        b = (rec.get("build_sha") or rec.get("commit") or
             (rec.get("metadata") or {}).get("build_sha"))
        if b:
            builds[str(b)[:10]] += 1
        a = (rec.get("kernel_arch") or rec.get("arch") or
             (rec.get("metadata") or {}).get("kernel_arch"))
        if a:
            archs[str(a)] += 1
        fp = rec.get("final_payload") or {}
        ch = (fp.get("choices") or [{}])[0]
        if (ch.get("logprobs") or {}).get("top_logprobs"):
            has_logprobs += 1
        if rec.get("metrics"):
            has_metrics += 1
    return {
        "n_records": n,
        "engine_versions": dict(versions),
        "build_shas": dict(builds),
        "kernel_archs": dict(archs),
        "records_with_logprobs": has_logprobs,
        "records_with_metrics": has_metrics,
        "is_v020_compatible": True,
    }


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
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--inspect", action="store_true",
                    help="print version summary only, no rewrite")
    args = ap.parse_args()
    records = load(args.src)
    summary = detect_version(records)
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if args.inspect:
        sys.exit(0)
    if not args.out:
        print("[v020_adapter] --out required (or use --inspect)", file=sys.stderr)
        sys.exit(2)
    primary_v = next(iter(summary["engine_versions"]), "unknown")
    primary_b = next(iter(summary["build_shas"]), "unknown")
    with open(args.out, "w") as f:
        for rec in records:
            rec["__capture_version__"] = {
                "engine_version": primary_v,
                "build_sha": primary_b,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[v020_adapter] wrote {len(records)} enriched records → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
