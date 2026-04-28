#!/usr/bin/env python3
# --ProtoAI-Bakari--
# METRICS_OK correctness gate captures both text+logprobs AND streaming perf metrics
# (TTFT, TPOT, ITL p50/p95/p99, PP TPS, TG TPS, wall) via SSE; non-streaming fallback
# still records wall + token counts so the LLM-test-metrics hook is satisfied.
#
# Capture a single prompt's completion + (when supported) top-K logprobs.
# Reads prompt JSONL line(s) on stdin, POSTs to BASELINE_ENDPOINT, appends
# result line to BASELINE_OUT under file lock.
#
# Env:
#   BASELINE_ENDPOINT   default http://10.255.255.11:8000   (cuda1 vLLM PP)
#   BASELINE_MODEL      default auto-detect via /v1/models  (uses first id)
#   BASELINE_OUT        default ./baseline.jsonl
#   BASELINE_LOGPROBS   default 20  (set 0 to disable; some MLX servers reject)
#   BASELINE_TIMEOUT    default 120 (seconds per request)
#   BASELINE_BACKEND    default completions  (or 'chat')
#   BASELINE_STREAM     default 1   (0 to disable; MLX servers without SSE will need 0)
import argparse, fcntl, json, os, sys, time, statistics, urllib.request, urllib.error

ENDPOINT = os.environ.get("BASELINE_ENDPOINT", "http://10.255.255.11:8000").rstrip("/")
MODEL = os.environ.get("BASELINE_MODEL", "")
OUT = os.environ.get("BASELINE_OUT", "baseline.jsonl")
LOGPROBS = int(os.environ.get("BASELINE_LOGPROBS", "20"))
TIMEOUT = int(os.environ.get("BASELINE_TIMEOUT", "120"))
BACKEND = os.environ.get("BASELINE_BACKEND", "completions")
STREAM = os.environ.get("BASELINE_STREAM", "1") not in ("0", "false", "no")

def detect_model():
    global MODEL
    if MODEL:
        return MODEL
    req = urllib.request.Request(ENDPOINT + "/v1/models")
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    MODEL = data["data"][0]["id"]
    return MODEL

def append_jsonl(path, obj):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with open(path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def post_unary(path, body):
    req = urllib.request.Request(
        ENDPOINT + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())

def post_stream(path, body):
    """SSE consumer. Returns (full_text, token_timestamps[], final_payload, prompt_tokens, completion_tokens)."""
    body = dict(body)
    body["stream"] = True
    if BACKEND != "chat":
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        ENDPOINT + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    full_text_parts = []
    token_ts = []
    final_payload = None
    prompt_tokens = None
    completion_tokens = None
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = (chunk.get("choices") or [{}])[0]
            delta = ch.get("delta") or {}
            text_piece = ch.get("text") or delta.get("content") or ""
            if text_piece:
                token_ts.append(time.time())
                full_text_parts.append(text_piece)
            if chunk.get("usage"):
                final_payload = chunk
                prompt_tokens = chunk["usage"].get("prompt_tokens")
                completion_tokens = chunk["usage"].get("completion_tokens")
    return "".join(full_text_parts), token_ts, final_payload, prompt_tokens, completion_tokens

def compute_metrics(t_send, t_first, t_end, prompt_tokens, completion_tokens, token_ts):
    wall = t_end - t_send
    ttft = (t_first - t_send) if t_first else None
    gen_time = (t_end - t_first) if t_first else None
    tpot = (gen_time / max(1, (completion_tokens or len(token_ts)) - 1)) if (gen_time and (completion_tokens or len(token_ts)) > 1) else None
    # ITL = inter-token latency between consecutive token events
    itls = []
    for i in range(1, len(token_ts)):
        itls.append(token_ts[i] - token_ts[i-1])
    itl_stats = {}
    if itls:
        itl_stats = {
            "p50": round(statistics.median(itls), 4),
            "p95": round(sorted(itls)[int(0.95 * (len(itls) - 1))], 4) if len(itls) > 1 else round(itls[0], 4),
            "p99": round(sorted(itls)[int(0.99 * (len(itls) - 1))], 4) if len(itls) > 1 else round(itls[0], 4),
            "mean": round(statistics.mean(itls), 4),
            "n": len(itls),
        }
    pp_tps = (prompt_tokens / ttft) if (prompt_tokens and ttft and ttft > 0) else None
    tg_tps = (completion_tokens / gen_time) if (completion_tokens and gen_time and gen_time > 0) else (
        (len(token_ts) / gen_time) if (gen_time and gen_time > 0 and token_ts) else None
    )
    return {
        "wall_s": round(wall, 4),
        "TTFT_s": round(ttft, 4) if ttft is not None else None,
        "TPOT_s": round(tpot, 4) if tpot is not None else None,
        "ITL": itl_stats,
        "PP_TPS": round(pp_tps, 2) if pp_tps is not None else None,
        "TG_TPS": round(tg_tps, 2) if tg_tps is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens or len(token_ts),
    }

def capture(prompt_obj):
    detect_model()
    body = {
        "model": MODEL,
        "max_tokens": prompt_obj.get("max_tokens", 64),
        "temperature": 0,
    }
    if prompt_obj.get("stop"):
        body["stop"] = prompt_obj["stop"]
    if BACKEND == "chat":
        path = "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": prompt_obj["prompt"]}]
    else:
        path = "/v1/completions"
        body["prompt"] = prompt_obj["prompt"]
        if LOGPROBS > 0:
            body["logprobs"] = LOGPROBS

    t_send = time.time()
    err = None
    text = ""
    final_payload = None
    metrics = None
    token_ts = []

    try:
        if STREAM:
            text, token_ts, final_payload, p_tok, c_tok = post_stream(path, body)
            t_first = token_ts[0] if token_ts else None
            t_end = time.time()
            metrics = compute_metrics(t_send, t_first, t_end, p_tok, c_tok, token_ts)
        else:
            resp = post_unary(path, body)
            t_end = time.time()
            ch = (resp.get("choices") or [{}])[0]
            text = ch.get("text") or (ch.get("message") or {}).get("content") or ""
            final_payload = resp
            usage = resp.get("usage") or {}
            metrics = {
                "wall_s": round(t_end - t_send, 4),
                "TTFT_s": None, "TPOT_s": None, "ITL": {}, "PP_TPS": None, "TG_TPS": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            }
    except urllib.error.HTTPError as e:
        err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}"
    except urllib.error.URLError as e:
        err = f"URL error: {e}"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    out = {
        "id": prompt_obj["id"],
        "category": prompt_obj.get("category"),
        "model": MODEL,
        "endpoint": ENDPOINT,
        "backend": BACKEND,
        "stream": STREAM,
        "ts": int(time.time()),
        "request_max_tokens": body["max_tokens"],
    }
    if err:
        out["error"] = err
    else:
        out["text"] = text
        if final_payload is not None:
            out["final_payload"] = final_payload
        out["metrics"] = metrics
    append_jsonl(OUT, out)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="if set, only process this id from stdin")
    args = ap.parse_args()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"skip bad line: {e}\n")
            continue
        if args.id and obj.get("id") != args.id:
            continue
        r = capture(obj)
        if "error" in r:
            sys.stderr.write(f"  [{r['id']}] ERR={r['error'][:120]}\n")
        else:
            m = r.get("metrics") or {}
            sys.stderr.write(f"  [{r['id']}] wall={m.get('wall_s')}s ttft={m.get('TTFT_s')} tpot={m.get('TPOT_s')} tg_tps={m.get('TG_TPS')} ctok={m.get('completion_tokens')}\n")

if __name__ == "__main__":
    main()
