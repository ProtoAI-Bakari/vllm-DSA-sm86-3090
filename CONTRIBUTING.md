# Contributing to vllm-DSA-sm86-3090

Thanks for considering a contribution. This is a research-systems engineering project porting Hopper-only DSA kernels to Ampere sm_86. We welcome help at every level.

## Tier system — what hardware do you need?

| Tier | Hardware | Examples of what you can do |
|---|---|---|
| **L0** | Any machine | Documentation, typo fixes, README polish, link audits, formatting |
| **L1** | Single 3090 (or compatible 24 GB Ampere) | Build the wheel, run unit tests on per-kernel fixtures, validate compile |
| **L2** | 1 node (2× 3090) | TP=2 forward-slice integration tests |
| **L3** | 1-2 nodes | Small-model load tests, partial vLLM request path |
| **L4** | Full cluster (8 nodes, 16× 3090) | Real DSV4 / GLM-5.1 token generation, the canonical correctness gate |
| **L5** | Full cluster + perf bench | Concurrency sweeps, NCCL fabric profiling |

L0 + L1 contributors are the most welcome — most of the documentation + numerics work happens at those tiers.

## Authorship and attribution

This project follows a non-standard attribution model.

**Project author:** ProtoAI-Bakari (Bakari McCoy / IntuitIntel LLC) holds primary authorship and the patent grant on novel sm_86 substitution patterns.

**LLM assistance:** Where AI agents (Claude Opus 4.7 etc.) produced material, the trailer reads:

```
Assisted-By: Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC<N>]
```

NOT `Co-Authored-By:`. The distinction is intentional — LLM contributions are assistance, not co-authorship.

**Inline file stamps:** Every modified or new `.cu` / `.py` / `.sh` / `.md` file should have a header comment naming the contributor:

```
// =============================================================================
// vllm-DSA-sm86-3090 — DeepSeek Sparse Attention backport for sm_86 (Ampere)
// Modified: 2026-XX-XX by --<your-handle>--
// Reason:   <short reason for the change>
// =============================================================================
```

## Commit message format

```
[lane: <lane-name>] <one-line summary>

Why: <root cause or motivation, 1-3 sentences>
What: <code changes, bulleted>
Test: <which tier passed, L1/L2/L3/L4/L5>
Numerics: <if applicable: top-1 X%, cosine X, perplexity Y>

Refs: vLLM upstream PR #40760, issue #X, kernel inventory line N
Assisted-By: Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC<N>]
```

The `Assisted-By:` trailer is added automatically by the pre-commit hook in `.githooks/prepare-commit-msg`. If the hook detects a `Co-Authored-By: Claude*` trailer, it rewrites it to `Assisted-By:`.

## Branch strategy

- `main` — protected, ships only after L4 + L5 pass
- `lane-cc<N>-*` — agent-owned working branches (CC0 merges)
- `dryrun-cc<N>` — test branches for skeleton drills (auto-pruned)
- `pr-<short-name>` — community contribution branches (PRs targeted at `main`)

## Test pyramid

Run the lowest tier you have hardware for. CC6 builds the harness; you run it.

```bash
# L1 unit (single 3090)
bash tests/unit/run_all.sh

# L2 forward slice (1 node)
bash tests/integration/run_l2.sh

# L4 full-cluster (8 nodes, drained)
bash tests/correctness/run_l4.sh
```

## Code of conduct

Be excellent. No model-shaming, no fork-shaming. If you find a kernel approach that works better than ours, open a PR — we'll gladly take it. The point is to make DSA-class models run on Ampere fleets at scale.

## Reporting bugs

Open an issue on https://github.com/ProtoAI-Bakari/vllm-DSA-sm86-3090/issues. Use the templates in `.github/ISSUE_TEMPLATE/`. For numerics drift, attach the offending prompt + your local hardware stamp (`nvidia-smi -L`).

## Questions?

ProtoAI-Bakari can be reached at prototypearchitect@gmail.com or via GitHub @ProtoAI-Bakari.
