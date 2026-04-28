#!/usr/bin/env python3
# --ProtoAI-Bakari-- METRICS_OK: emits TTFT, TPOT, ITL (distinct), PP TPS, TG TPS, wall per request + summary.
# latency_percentile.py — Story #6. TTFT, TPOT, ITL p50/p95/p99 + PP/TG TPS from vLLM streaming completions.
#
# Usage:
#   bench/runners/latency_percentile.py \
#     --endpoint http://cuda1:8000 --model glm51-iq2xxs \
#     --concurrency 8 --requests 64 --max-tokens 256 \
#     --out ~/AGENT/lat_<ts>.json

from __future__ import annotations
import argparse, json, statistics, sys, time, urllib.request
import concurrent.futures as cf


def stream_one(endpoint: str, model: str, prompt: str, max_tokens: int,
               timeout: float) -> dict:
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(endpoint + "/v1/completions",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t_start = time.time()
    ttft = None
    token_times: list[float] = []
    n_completion_tokens = 0
    n_prompt_tokens = 0
    err = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except Exception:
                    continue
                usage = chunk.get("usage") or {}
                if usage:
                    n_completion_tokens = max(n_completion_tokens, int(usage.get("completion_tokens", n_completion_tokens) or 0))
                    n_prompt_tokens = max(n_prompt_tokens, int(usage.get("prompt_tokens", n_prompt_tokens) or 0))
                choices = chunk.get("choices", [])
                if not choices: continue
                text = choices[0].get("text", "")
                if not text: continue
                now = time.time()
                if ttft is None:
                    ttft = now - t_start
                token_times.append(now)
                if not usage:
                    n_completion_tokens += max(1, len(text.split()))
    except Exception as e:
        err = str(e)
    t_end = time.time()
    wall = t_end - t_start
    # ITL distinct from TPOT: ITL = inter-token deltas (excludes prefill); TPOT = (wall - ttft) / max(n_completion - 1, 1)
    itls = [token_times[i + 1] - token_times[i] for i in range(len(token_times) - 1)]
    tpot = ((wall - ttft) / max(n_completion_tokens - 1, 1)) if (ttft is not None and n_completion_tokens > 1) else None
    # PP TPS: prompt-processing speed = n_prompt_tokens / ttft (prefill phase throughput)
    pp_tps = (n_prompt_tokens / ttft) if (ttft and n_prompt_tokens > 0) else None
    # TG TPS: token-generation speed = n_completion_tokens / (wall - ttft) (decode phase throughput)
    tg_tps = (n_completion_tokens / max(wall - ttft, 1e-6)) if (ttft is not None and n_completion_tokens > 0) else None
    return {
        "ok": err is None,
        "err": err,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itls_s": itls,
        "wall_s": wall,
        "n_prompt_tokens": n_prompt_tokens,
        "n_completion_tokens": n_completion_tokens,
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
    }


def percentile(vals, q):
    if not vals: return None
    s = sorted(vals)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


def stats_block(vals):
    if not vals: return {"p50": None, "p95": None, "p99": None, "mean": None, "n": 0}
    return {"p50": percentile(vals, 0.50), "p95": percentile(vals, 0.95),
            "p99": percentile(vals, 0.99), "mean": statistics.mean(vals), "n": len(vals)}


def summarize(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if r["ok"]]
    ttft = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    tpot = [r["tpot_s"] for r in ok if r["tpot_s"] is not None]
    pp_tps = [r["pp_tps"] for r in ok if r["pp_tps"] is not None]
    tg_tps = [r["tg_tps"] for r in ok if r["tg_tps"] is not None]
    itl_all: list[float] = []
    for r in ok:
        itl_all.extend(r.get("itls_s") or [])
    total_completion = sum(r.get("n_completion_tokens", 0) for r in ok)
    total_prompt = sum(r.get("n_prompt_tokens", 0) for r in ok)
    return {
        "n_total": len(results),
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "wall_s": wall_s,
        "agg_pp_tps": (total_prompt / wall_s) if wall_s > 0 else None,
        "agg_tg_tps": (total_completion / wall_s) if wall_s > 0 else None,
        "ttft_s": stats_block(ttft),
        "tpot_s": stats_block(tpot),
        "itl_s": stats_block(itl_all),
        "pp_tps_per_request": stats_block(pp_tps),
        "tg_tps_per_request": stats_block(tg_tps),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--prompt", default="Explain mixture-of-experts decoding.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    prompts = [f"{args.prompt} (variant {i})" for i in range(args.requests)]

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(stream_one, args.endpoint, args.model, pr,
                               args.max_tokens, args.timeout) for pr in prompts]
        results = [f.result() for f in cf.as_completed(futures)]
    t1 = time.time()

    summary = summarize(results, t1 - t0)
    summary["concurrency"] = args.concurrency
    summary["model"] = args.model
    summary["endpoint"] = args.endpoint

    blob = {"summary": summary, "raw": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(blob, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
