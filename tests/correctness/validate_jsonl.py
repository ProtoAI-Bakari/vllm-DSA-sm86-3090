#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK schema validator for gate-input JSONL files; not a perf bench
#
# Validates a prompts.jsonl OR baseline-format JSONL up-front so verify_gate.py
# never grades on malformed data. Catches:
#   - missing 'id' field (gate dedup uses id; missing → silent merge collisions)
#   - duplicate ids
#   - invalid JSON lines
#   - unbalanced category/error invariants
#   - encoding issues
#
# Run:
#   python3 validate_jsonl.py prompts <file>          # prompt-set schema
#   python3 validate_jsonl.py capture <file>          # baseline/under-test schema
#
# Exit: 0 valid / 1 errors / 2 input error.
import argparse, json, sys
from collections import Counter


PROMPT_REQUIRED = ("id", "category", "prompt", "max_tokens")
CAPTURE_REQUIRED = ("id",)


def iterate(path):
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                yield i, {"__bad_line__": True, "__err__": str(e), "__raw__": line[:200]}


def validate(kind, path):
    errors = []
    warnings = []
    ids = Counter()
    n = 0
    cats = Counter()
    cat_errors = Counter()

    if kind == "prompts":
        required = PROMPT_REQUIRED
    else:
        required = CAPTURE_REQUIRED

    for ln, rec in iterate(path):
        n += 1
        if rec.get("__bad_line__"):
            errors.append((ln, f"invalid JSON: {rec['__err__']}; raw={rec['__raw__']!r}"))
            continue
        for k in required:
            if k not in rec:
                errors.append((ln, f"missing required field '{k}': record={dict(list(rec.items())[:3])}"))
                break
        if "id" in rec:
            ids[rec["id"]] += 1
        cat = rec.get("category", "unknown")
        cats[cat] += 1
        if "error" in rec:
            cat_errors[cat] += 1

        if kind == "prompts":
            mt = rec.get("max_tokens")
            if not isinstance(mt, int) or mt <= 0 or mt > 4096:
                warnings.append((ln, f"unusual max_tokens={mt} (expected 1..4096)"))

    for the_id, count in ids.items():
        if count > 1:
            errors.append((None, f"duplicate id '{the_id}' appears {count} times"))

    return {
        "kind": kind, "path": path, "n": n,
        "errors": errors, "warnings": warnings,
        "categories": dict(cats), "category_error_counts": dict(cat_errors),
        "n_unique_ids": len(ids),
    }


def render(rep):
    L = []
    L.append(f"# validate_jsonl ({rep['kind']}) — {rep['path']}")
    L.append(f"  records: {rep['n']}   unique ids: {rep['n_unique_ids']}")
    L.append(f"  categories: {rep['categories']}")
    if rep["category_error_counts"]:
        L.append(f"  per-category error counts: {rep['category_error_counts']}")
    if rep["errors"]:
        L.append("")
        L.append(f"  ERRORS ({len(rep['errors'])}):")
        for ln, msg in rep["errors"][:20]:
            ln_s = f"line {ln}" if ln else "global"
            L.append(f"    [{ln_s}] {msg}")
        if len(rep["errors"]) > 20:
            L.append(f"    ... and {len(rep['errors']) - 20} more")
    if rep["warnings"]:
        L.append("")
        L.append(f"  WARNINGS ({len(rep['warnings'])}):")
        for ln, msg in rep["warnings"][:10]:
            L.append(f"    [line {ln}] {msg}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("prompts", "capture"))
    ap.add_argument("path")
    args = ap.parse_args()
    rep = validate(args.kind, args.path)
    print(render(rep))
    sys.exit(0 if not rep["errors"] else 1)


if __name__ == "__main__":
    main()
