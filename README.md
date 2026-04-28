# vllm-DSA-sm86-3090

**DeepSeek Sparse Attention (DSA) backport for NVIDIA Ampere (sm_86 / RTX 3090).**

DSV4-Flash and GLM-5.1-IQ2XXS depend on DSA — DeepSeek's sparse-attention architecture introduced in vLLM 0.20.0 (PR #40760). The DSA kernels live inside `vllm/_C.abi3.so` and hardware-gate Hopper-only (TMA + WGMMA + cp.async.bulk + PDL). On Ampere (sm_86) cards like the RTX 3090, the server boots and `/v1/models` returns 200, but `/v1/completions` returns 500 because the assertion fires inside compiled CUDA we can't reach with Python patches.

This repo rebuilds those kernels with sm_86 fallback paths (TMA → `cp.async.cg`, WGMMA → `mma.sync.aligned.m16n8k16`, etc.) so DSA-architecture models — DSV4 + GLM-5.1 + future SOTA — can run on RTX 3090 fleets.

## Status

🚧 Active development. See `docs/HISTORY.md` for the path that got us here (17-patch cascade, all forensics).

## Quick start (placeholder — populated by CC8 once first build ships)

```bash
# clone + build sm_86 fallback wheel (reproducible)
git clone https://github.com/ProtoAI-Bakari/vllm-DSA-sm86-3090.git
cd vllm-DSA-sm86-3090
bash scripts/build_vllm_dsa_sm86.sh
# install on a single 3090
pip install dist/vllm-dsa-sm86-*.whl
# launch DSV4-Flash-FP8 on TP=2 × EP=8 (16x 3090 cluster)
bash scripts/launch_dsv4_tp2_ep8.sh
```

## Numerics retention

Acceptance gate (CC6 enforces): top-1 token agreement ≥98% vs llama.cpp PP baseline, logit cosine ≥0.97, perplexity drop ≤2% on MMLU subset.

| Model | Topology | conc=1 | aggregate | top-1 vs PP | perplexity Δ |
|---|---|---|---|---|---|
| DSV4-Flash-FP8 | PP=2 × TP=8 | _pending_ | _pending_ | _pending_ | _pending_ |
| DSV4-Flash-FP8 | TP=2 × EP=8 | _pending_ | _pending_ | _pending_ | _pending_ |
| GLM-5.1-IQ2XXS | TP=2 × EP=8 | _pending_ | _pending_ | _pending_ | _pending_ |

## Performance tier scale

**Aggregate (multi-conc):** Bronze ≥100 / Silver ≥300 / Gold ≥500 / Platinum ≥700 / Diamond ≥900 / Cosmic ≥1000 t/s
**conc=1 (single-user, properly tuned):** Bronze 25 / Silver 35 / Gold 55 / Platinum 65 / Diamond 75 t/s

## Hardware tested

- 16× NVIDIA RTX 3090 (sm_86, 24 GB each, 384 GB aggregate VRAM) on 100 GbE RoCEv2 fabric
- Single 3090 contributors welcome at L0/L1 test tiers (see `CONTRIBUTING.md`)

## Known limitations

- BF16 fallback for FP8 [128,128] block-scaled GEMM is ~2× slower than native Hopper FP8 — fundamental, not a bug
- DSV4 ceiling on rebuilt sm_86 stack is physics-bound at ~10-20 t/s aggregate; GLM-5.1 reaches higher because more native-FP8 paths survive the substitution
- TP=16 may require model-side `o_groups` relaxation (HIGH-RISK lane); PP=2+TP=8 = verified boot fallback

## License

Apache-2.0 (matches upstream `vllm-project/vllm`). See `LICENSE`.

## Authorship

Authored by **ProtoAI-Bakari** (Bakari McCoy / IntuitIntel LLC).
With Assistance by **Claude Opus 4.7** (`claude-opus-4-7`) [1M ctx, max effort].
Multi-agent orchestration framework (CC0 lead + CC1-CC9 workers) detailed in `docs/HISTORY.md`.

## Contributing

See `CONTRIBUTING.md`. Tier system: L0/L1 contributors test on a single 3090; L4/L5 require the full 16-GPU cluster.

## Related projects (ProtoAI)

- `vllm-asahi-vulkan-0.17.1` — vLLM on Asahi Linux + Vulkan (M1/M2 Macs)
- `vllm-protoai-3090-turboquant-sm86` — earlier turboquant scope, superseded by this repo
- `vllm-metal` — vLLM-Metal for Apple Silicon
