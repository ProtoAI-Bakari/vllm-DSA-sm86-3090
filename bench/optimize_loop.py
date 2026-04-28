#!/usr/bin/env python3
# --ProtoAI-Bakari--
# optimize_loop.py — Story #9. Given bench output classified at tier T, propose next-step
# optimization candidates per corpus §8.2 / EPIC 4 ladder.
#
# Usage:
#   optimize_loop.py --tier-classify ~/AGENT/tier_<ts>.json [--gpu-util ~/AGENT/gpu_util_<ts>.csv] [--kv ~/AGENT/kv_<ts>.jsonl]
#
# Output: ranked list of candidate optimizations with rationale + expected uplift.

from __future__ import annotations
import argparse, csv, json, statistics, sys
from typing import Optional


# Optimization library — ordered by ROI on Ampere TP=2 x EP=8.
OPT_LIBRARY = [
    {
        "name": "fp8_kv_cache",
        "gates": "kv_dtype != fp8",
        "expected_uplift_pct": 5,
        "tier_relevance": ["below_bronze", "bronze", "silver"],
        "blast_radius": "low (~1% perplexity tax)",
        "rationale": "halves KV bandwidth, doubles effective KV slots; sm_86 native.",
    },
    {
        "name": "max_num_batched_tokens_sweep",
        "gates": "single fixed value used",
        "expected_uplift_pct": 8,
        "tier_relevance": ["below_bronze", "bronze", "silver"],
        "blast_radius": "low (no model changes)",
        "rationale": "chunked-prefill knob, batch coalescing under conc>=4.",
    },
    {
        "name": "attention_prefix_cache",
        "gates": "prefix_cache disabled",
        "expected_uplift_pct": 10,
        "tier_relevance": ["below_bronze", "bronze"],
        "blast_radius": "low",
        "rationale": "shared-prefix workloads (system prompt) reuse KV across requests.",
    },
    {
        "name": "int4_awq_marlin_sm86",
        "gates": "FP8 emulation path on Ampere (DSV4)",
        "expected_uplift_pct": 200,
        "tier_relevance": ["below_bronze", "bronze"],
        "blast_radius": "high (numerics gate ≤2%)",
        "rationale": "274GB FP8 -> 68GB INT4 across 16 ranks = 4.3GB/rank; mem-BW ceiling rises 4x.",
    },
    {
        "name": "mtp_spec_decode",
        "gates": "MTP head not wired",
        "expected_uplift_pct": 75,
        "tier_relevance": ["bronze", "silver", "gold"],
        "blast_radius": "med (deterministic temp=0 verification gate)",
        "rationale": "DSV4 native multi-token prediction head; +1.5-2x free if wired.",
    },
    {
        "name": "int8_kv_cache",
        "gates": "fp8_kv already on; further halving",
        "expected_uplift_pct": 8,
        "tier_relevance": ["silver", "gold", "platinum"],
        "blast_radius": "low (~1% perplexity)",
        "rationale": "sm_86 native; halves again on top of fp8 (effective 4x slot count).",
    },
    {
        "name": "eplb_load_balancer",
        "gates": "expert load skew >10%",
        "expected_uplift_pct": 15,
        "tier_relevance": ["silver", "gold"],
        "blast_radius": "med (reroutes experts mid-batch)",
        "rationale": "keeps all 8 active experts hot under bursty traffic; corpus §EPIC 3.3.",
    },
    {
        "name": "fused_moe_kernels_sm86_backport",
        "gates": "vLLM sm_90 fused MoE not yet ported",
        "expected_uplift_pct": 20,
        "tier_relevance": ["gold", "platinum"],
        "blast_radius": "high (kernel surgery)",
        "rationale": "merges expert dispatch + GEMM + reduce into one kernel; 1.2-1.4x.",
    },
    {
        "name": "rocev2_p2p_driver_patch",
        "gates": "iperf3 mesh < 50 Gbps p95",
        "expected_uplift_pct": 25,
        "tier_relevance": ["gold", "platinum", "diamond"],
        "blast_radius": "high (driver-level)",
        "rationale": "removes user/kernel copy on EP all-to-all; aspirational, AO2 audit flag.",
    },
    {
        "name": "eagle3_speculative_decode",
        "gates": "draft head not configured",
        "expected_uplift_pct": 50,
        "tier_relevance": ["platinum", "diamond"],
        "blast_radius": "high (separate draft model + verify)",
        "rationale": "Diamond conc=1 (75 t/s) requires this layer per corpus §8.4.",
    },
]


def parse_gpu_util(path: Optional[str]) -> Optional[float]:
    if not path: return None
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        return None
    vals = []
    for r in rows:
        try:
            vals.append(float(r["util_gpu"]))
        except (KeyError, ValueError):
            continue
    return statistics.mean(vals) if vals else None


def parse_kv(path: Optional[str]) -> Optional[float]:
    if not path: return None
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return None
    vals = [r.get("vllm:gpu_cache_usage_perc") for r in rows]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.mean(vals) if vals else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier-classify", required=True)
    p.add_argument("--gpu-util")
    p.add_argument("--kv")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(args.tier_classify) as f:
        tier = json.load(f)

    agg_tier = tier.get("aggregate", {}).get("tier", "below_bronze")
    conc1_tier = tier.get("conc1", {}).get("tier", "below_bronze")

    gpu_util_mean = parse_gpu_util(args.gpu_util)
    kv_mean = parse_kv(args.kv)

    candidates = []
    for opt in OPT_LIBRARY:
        if agg_tier in opt["tier_relevance"] or conc1_tier in opt["tier_relevance"]:
            candidates.append(opt)

    # Diagnostic-aware filtering
    diagnostics = {
        "gpu_util_mean": gpu_util_mean,
        "kv_usage_mean_perc": kv_mean,
        "agg_tier": agg_tier,
        "conc1_tier": conc1_tier,
    }
    if gpu_util_mean is not None and gpu_util_mean < 50:
        for c in candidates:
            if c["name"] in ("rocev2_p2p_driver_patch", "max_num_batched_tokens_sweep"):
                c["priority_boost"] = "fabric/batch limited (gpu_util < 50%)"

    out = {
        "diagnostics": diagnostics,
        "candidates": sorted(candidates, key=lambda x: -x["expected_uplift_pct"]),
    }

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
