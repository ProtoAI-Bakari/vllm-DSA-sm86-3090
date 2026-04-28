#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK correctness gate input adapter; not a perf bench
# Convert CC9-shipped profile JSON dumps (eg
#   /repo/models/RUN/PROFILE_RESULTS/<model>__<ts>.json
# ) into baseline-format JSONL the gate (verify_gate.py / perplexity_full.py /
# score_long_ctx.py) can consume.
#
# Heuristic format detection (no spec — accept what comes):
#   1. JSON list of fixture dicts at top level
#   2. JSON dict with "fixtures" | "results" | "tests" | "items" key
#   3. JSON dict where each top-level key maps to a fixture-like dict (id keyed)
#
# Per-fixture key heuristics, in priority:
#   id            : id | prompt_id | name | key
#   prompt        : prompt | input | request.prompt | request.messages[-1].content
#   text          : completion | output | response.text | response.choices[0].text
#                   | response.choices[0].message.content
#   logprobs      : response.choices[0].logprobs (passed through verbatim)
#   error         : error | failure | exception | non-200 status_code
#   category      : category | tag | suite | "unknown"
#
# Usage:
#   python3 profile_to_jsonl.py --in <profile.json> --out <baseline.jsonl> [--integ <label>]
#   python3 profile_to_jsonl.py --in - --out -                    # stdin/stdout
#   python3 profile_to_jsonl.py --in <profile.json> --schema list # force a schema
import argparse, json, os, sys

CANDIDATE_LIST_KEYS = ("fixtures", "results", "tests", "items", "completions", "captures")


def find_fixture_iter(blob, schema=None):
    if schema == "list" and isinstance(blob, list):
        return iter(blob)
    if isinstance(blob, list):
        return iter(blob)
    if isinstance(blob, dict):
        for k in CANDIDATE_LIST_KEYS:
            if k in blob and isinstance(blob[k], list):
                return iter(blob[k])
        # dict-of-dicts fallback (id-keyed)
        if all(isinstance(v, dict) for v in blob.values()):
            def _gen():
                for k, v in blob.items():
                    if "id" not in v:
                        v = dict(v); v["id"] = k
                    yield v
            return _gen()
    raise ValueError(f"unable to locate fixture list in profile (top-level type={type(blob).__name__})")


def _get_path(d, path, default=None):
    cur = d
    for part in path:
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)] if part != "-1" else cur[-1]
            except (ValueError, IndexError):
                return default
        else:
            return default
        if cur is None:
            return default
    return cur


def extract_id(fix, fallback_idx):
    for k in ("id", "prompt_id", "name", "key", "test_id"):
        if k in fix and fix[k]:
            return str(fix[k])
    return f"profile_{fallback_idx:04d}"


def extract_prompt(fix):
    for k in ("prompt", "input", "query"):
        if k in fix and isinstance(fix[k], str):
            return fix[k]
    msg = _get_path(fix, ["request", "messages", "-1", "content"])
    if isinstance(msg, str):
        return msg
    p = _get_path(fix, ["request", "prompt"])
    if isinstance(p, str):
        return p
    return ""


def extract_text(fix):
    for k in ("completion", "output", "text", "response_text"):
        if k in fix and isinstance(fix[k], str):
            return fix[k]
    t = _get_path(fix, ["response", "choices", "0", "text"])
    if isinstance(t, str):
        return t
    t = _get_path(fix, ["response", "choices", "0", "message", "content"])
    if isinstance(t, str):
        return t
    return ""


def extract_logprobs_payload(fix):
    payload = fix.get("response") if isinstance(fix.get("response"), dict) else None
    if payload and isinstance(payload.get("choices"), list):
        return payload
    return None


def extract_error(fix):
    for k in ("error", "failure", "exception"):
        if fix.get(k):
            return str(fix[k])[:400]
    sc = fix.get("status_code") or _get_path(fix, ["response", "status_code"])
    if isinstance(sc, int) and sc != 200:
        return f"HTTP {sc}"
    return None


def convert_fixture(fix, idx):
    out = {
        "id": extract_id(fix, idx),
        "category": fix.get("category") or fix.get("tag") or fix.get("suite") or "unknown",
    }
    text = extract_text(fix)
    err = extract_error(fix)
    if err:
        out["error"] = err
    if text:
        out["text"] = text
    pay = extract_logprobs_payload(fix)
    if pay is not None:
        out["final_payload"] = pay
    if "expected_letter" in fix:
        out["expected_letter"] = fix["expected_letter"]
    if "subject" in fix:
        out["subject"] = fix["subject"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", default=None, help="force schema: list|dict-fixtures|dict-id")
    ap.add_argument("--integ", default=None, help="label written into output for traceability")
    args = ap.parse_args()

    src = sys.stdin if args.src == "-" else open(args.src)
    blob = json.load(src)
    if args.src != "-":
        src.close()

    fixtures = list(find_fixture_iter(blob, schema=args.schema))
    out_f = sys.stdout if args.out == "-" else open(args.out, "w")
    n_emit = 0
    n_err = 0
    for i, fix in enumerate(fixtures):
        if not isinstance(fix, dict):
            sys.stderr.write(f"  skip non-dict fixture at idx {i}\n")
            continue
        rec = convert_fixture(fix, i)
        if args.integ:
            rec["integ"] = args.integ
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_emit += 1
        if "error" in rec:
            n_err += 1
    if args.out != "-":
        out_f.close()

    sys.stderr.write(f"[profile_to_jsonl] emitted {n_emit} fixtures ({n_err} with errors) → {args.out}\n")


if __name__ == "__main__":
    main()
