# CC2 Story Backlog — Build Infrastructure
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC2]**

**Pruned 2026-04-28 per CC0 retired-audit:** every story traces directly to the goal of DSV4 + GLM-5.1 coherent tokens at perf tier. Story #8 (wheel signing + post-Gold OSS reproducibility) cut — defer to post-coherent-token launch. The free deterministic flags `SOURCE_DATE_EPOCH` + `PYTHONHASHSEED` already baked into `build_vllm_dsa_sm86.sh` (they are zero-cost, harmless, and aid debug repro).

| # | Story | Complexity | Output | Status | Rationale |
|---|---|---|---|---|---|
| 1 | Clone vllm at SHA matching .venv-vllm_0_20_0_fresh into /repo/INSTALLERS/vllm.dsa.sm86/ | S | clone present + SHA recorded | shipped (in build script) | Source tree is the medium for sm_86 kernel ports — without it CC3/CC4/CC5 cannot land DSV4/GLM-5.1 unblock patches. -> unblocks DSV4/GLM-5.1 token gen. |
| 2 | Apply P6+P7+P8 baseline patches (P6/P8 vendor-tree commit, P7 sp post-install) | S | patches applied + git committed inside vendor tree | shipped | Without P6/P7/P8 baseline the rebuilt wheel still hits the UE8M0 + W8A8 walls already cleared on PR40760 venv — direct prerequisite for first non-500 token. -> unblocks DSV4/GLM-5.1 token gen. |
| 3 | cmake -DTORCH_CUDA_ARCH_LIST="8.6" config + smoke build (no DSA changes) | M | dist/vllm-baseline-sm86-*.whl + Qwen3.5 verify HTTP 200 | in CC9 exec | sm_86 wheel is the only artifact CC3/CC4/CC5 ports drop into; smoke verification proves the harness before kernel surgery. -> unblocks DSV4/GLM-5.1 token gen. |
| 4 | ABI lock-in — pin torch / cuda / nvcc / triton + reproducibility lockfile | M | scripts/build_env.lock + scripts/verify_env.sh | shipped | Drift in torch/cuda/nvcc broke PR40760 build twice; lockfile prevents CC3-CC5 chasing phantom kernel bugs caused by ABI skew. -> unblocks DSV4/GLM-5.1 token gen. |
| 5 | ccache + sccache integration for fast iteration | S | scripts/build_vllm_dsa_sm86.sh w/ cache config | shipped (USE_CCACHE/USE_SCCACHE knobs) | CC3-CC5 will iterate single .cu kernels; full 30-60min recompile per iter is fatal to wall-clock. ccache makes each iter 1-3 min. -> unblocks DSV4/GLM-5.1 token gen. |
| 6 | 0.19/0.20 dual-track build matrix — same patches against both vllm versions | M | scripts/build_matrix.sh + delta report | next | If 0.20 baseline fails Qwen3.5 verify but 0.19 holds, we route DSV4 path through 0.19; without dual-track we lose 1 day debugging the wrong version. -> unblocks DSV4/GLM-5.1 token gen. |
| 7 | Per-kernel rebuild orchestration — script that takes a single .cu patch + does incremental rebuild | L | scripts/incr_rebuild.sh | next | Iteration speed for CC3 sparse_attn_indexer port (12-16 cycles expected). Without this: 8h compile cost on a 1-day kernel. -> unblocks DSV4/GLM-5.1 token gen. |
| ~~8~~ | ~~Wheel signing + reproducibility (deterministic build flags)~~ | ~~S~~ | ~~scripts/build_reproducible.sh~~ | **CUT 2026-04-28** | Post-Gold OSS — does not unblock first coherent token. Free SOURCE_DATE_EPOCH/PYTHONHASHSEED already baked into build_vllm_dsa_sm86.sh as a no-cost side benefit; full signing + repro hashing deferred to EPIC 5 OSS launch. |
| 9 | Cluster-side install wrapper used by CC7 fanout — must work on cuda1-8 fresh | M | scripts/install_node.sh | next | CC7 fanout needs identical install on every node; ad-hoc rsync of wheel + venv breaks 1 in 8 nodes silently. Wrapper enforces parity. -> unblocks DSV4/GLM-5.1 token gen. |

**Workflow:** ship story → cc_git push → bridge milestone → IMMEDIATELY pick next. Recompile cycles between stories 3-7 give you idle compute time — use it for next-story authoring.

**Wall:** 8-12h. Critical path for CC3/CC4/CC5 unlock at story 3 + 7.

## Status snapshot 2026-04-28 03:55Z

- Stories 1+2 shipped via build_vllm_dsa_sm86.sh refactor (vendor-tree P6/P8 + sp P7).
- Story 3 in CC9 execution queue (approval id=1 APPROVED).
- Stories 4+5 shipped via build_env.lock + verify_env.sh + USE_CCACHE/USE_SCCACHE knobs.
- Story 6 next (build_matrix.sh dual 0.19/0.20).
- Story 7 next-after-6 (incr_rebuild.sh).
- Story 9 last-on-deck (install_node.sh; CC7 owns fanout, CC2 authors install primitive).
