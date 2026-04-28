# Story 2 — `MoEPrepareAndFinalizeNoDPEPModular` Audit for DSA EP=8 Dispatch

**Track:** A (MoE-DSA dispatch — unblocks TP=2×EP=8 topology)
**Lane:** `lane-cc5-moe-dsa`
**Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC5]**

## 1. Why this class

`MoEPrepareAndFinalizeNoDPEPModular` is vLLM's MoE prepare/finalize stage
for the **no-DP, EP-only** topology — the exact shape CC8 is targeting
(`TP=2 × EP=8 = 16 ranks`, no data-parallel replication of experts). It
sits between the **router output** and the **expert-shard GEMM kernels**:

```
hidden_state ── router/indexer ── topk(experts) ── PREPARE ── expert GEMM ── FINALIZE ── reduced_hidden
                                                  ▲                          ▲
                                                  │                          │
                       This audit ───────────────┘──────────────────────────┘
```

Two responsibilities:

1. **PREPARE** — gather tokens routed to *local* experts (this rank holds
   `n_total_experts / EP` of them; under EP=8 with 256 experts → 32 local
   experts), pad to a Marlin/cutlass-friendly tile shape, build the
   per-expert offset table.
2. **FINALIZE** — scatter expert outputs back into per-token slots,
   weighted by the router's top-k probabilities, sum across the k slots.

DSA replaces the **router** with the **Lightning Indexer's top-k pick**
(not standard learned router). The downstream PREPARE/FINALIZE shapes are
the same contract — but only **if Lightning Indexer's output schema
matches the router's expected `(topk_ids, topk_weights)` contract**.

That match is what this audit interrogates.

## 2. The shape contract under DSV4 / GLM-5.1

DSV4-Flash-FP8 + GLM-5.1-IQ2XXS share these MoE invariants:

| Invariant | Value | Source |
|---|---|---|
| Total experts (`E`) | 256 | DSV4 + GLM-5.1 configs |
| Active per token (`k`) | 8 | DSV4 + GLM-5.1 configs |
| Hidden size (`H`) | 7168 | DSV4 config |
| Expert intermediate (`I`) | 2048 | DSV4 config |
| `o_groups` (output partition) | 8 | DSV4 model invariant (breaks TP=16) |
| EP factor (planned) | 8 | CC8 target |
| Local experts per rank | 32 | `E / EP = 256 / 8` |

### 2.1 PREPARE input contract

vLLM expects (per-rank):

| Tensor | Shape | dtype | Provenance |
|---|---|---|---|
| `hidden_states` | `[T, H]` | bf16 | residual stream |
| `topk_ids` | `[T, k]` | int32 | **router** output (DSA: **Lightning Indexer top-k**) |
| `topk_weights` | `[T, k]` | bf16/fp32 | router softmax weights |
| `expert_map` | `[E]` (optional) | int32 | maps global expert id → local slot or `-1` |

`T` = `tokens_per_rank * tp` (varies with prefill vs decode).

### 2.2 PREPARE output contract (what expert GEMMs consume)

| Tensor | Shape | dtype | Notes |
|---|---|---|---|
| `gathered_tokens` | `[sum_local_e M_e, H]` | bf16 | concatenated, expert-major |
| `expert_offsets` | `[local_E + 1]` | int32 | CSR-style start per expert |
| `expert_ids_inv` | `[sum_local_e M_e]` | int32 | back-pointer for FINALIZE |
| `topk_weights_local` | `[sum_local_e M_e]` | bf16/fp32 | the k-weight that picked this token |

`M_e` = number of tokens this rank will run through expert `e`. Under
EP=8 with k=8, **expected `M_e ≈ T * 8 / 256 = T / 32`** in a balanced
load. Under skewed routing (the EPLB problem from Story 7), `M_e` can
spike 5-10× for hot experts.

## 3. Where DSA breaks the standard contract — the four risks

### Risk 1 — Lightning Indexer output dtype / layout

Lightning Indexer (the DSA-introduced top-k mechanism) lives in the
sparse-attention compiled unit. Its top-k output **may** ship as:

- `int64` indices (vLLM router emits `int32`)
- Soft-routing weights with **non-softmax** normalization
- Indices into a *sub-token* granularity (per query-head, not per-token)

If any of those hit `MoEPrepareAndFinalizeNoDPEPModular`, the gather
either silently miscounts (wrong dtype reinterpretation) or asserts on
shape mismatch.

**Patch surface (Track A Story 3):** insert a coercion stage between
Lightning Indexer's output and the prepare module input —
`csrc_patches/moe_dispatch_dsa_sm86.cu` will own the int64→int32 cast +
softmax-normalize step.

### Risk 2 — `o_groups=8` invariant breaks TP=16

DSV4's expert `w13` weight is sharded with `o_groups=8`. Standard TP=16
sharding asks for `o_groups % tp == 0` → `8 % 16 ≠ 0`, raises
`ZeroDivisionError` or `shape '[64,4,128]' invalid for size 65536`.

Under TP=2×EP=8, `tp=2` divides `o_groups=8` cleanly. **EP=8 topology
sidesteps this entirely**, which is one of the stronger reasons to ship
EP=8 over TP=16. Story 6 (HIGH-RISK isolated lane) attempts the
relaxation only after PP=2+TP=8 ships clean.

### Risk 3 — FP8 [128,128] block-scale path inside expert GEMM

The PREPARE module hands `gathered_tokens` to the expert GEMM. On
DSV4-FP8 the GEMM is DEEP_GEMM (Hopper-only). On Ampere, vLLM falls
back to BF16 emulation OR — once Story 5 lands — to **AWQ-Marlin INT4
GEMM** (Story 1 probe verifies kernel availability).

The PREPARE module itself is dtype-agnostic, so this risk does **not**
require a PREPARE patch — it lives in the GEMM dispatch (Story 4
covers AWQ-Marlin sm_86 backport).

### Risk 4 — Lightning Indexer dispatches at sub-token granularity

DSA's "sparse attention indexer" picks top-k *attention* contributors.
Whether those map 1:1 to MoE expert dispatch or whether DSV4 routes
per-head independently is a config detail buried in:

```
vllm/v1/attention/backends/mla/flashmla_sparse.py
vllm/model_executor/models/deepseek_v4.py
```

If per-head, then `topk_ids` shape becomes `[T, n_heads, k]` and the
PREPARE module's `[T, k]` assumption breaks at the gather step.

**Audit action:** CC1's kernel inventory should document which
indexer-output shape DSV4 emits. If per-head, add a `flatten(1,2)` +
`/n_heads` weight-rescale before PREPARE.

## 4. Symbol-level audit checklist (post-CC2-rebuild)

When CC2's `cmake -DTORCH_CUDA_ARCH_LIST="8.6"` rebuild lands, walk this
list with `nm` / `objdump` / Python `dir()`:

- [ ] `vllm.model_executor.layers.fused_moe.modular_kernel.MoEPrepareAndFinalizeNoDPEPModular` resolves at module import on cuda5
- [ ] `MoEPrepareAndFinalizeNoDPEPModular.prepare()` accepts `topk_ids: int32` (not int64) — coerce upstream if not
- [ ] `expert_map` parameter handles `local_E=32` slot mapping
- [ ] No `assert capability.major == 9` lurking inside the module
- [ ] `FINALIZE` accepts `expert_outputs: list[Tensor]` of length `local_E=32`
- [ ] All-to-all under EP=8 uses NCCL `alltoallv` (not Hopper PDL chained launches)
- [ ] `cudaStreamSynchronize` between PREPARE and expert-GEMM does not stall
      decode below 25 t/s conc=1 (Story 9 integration test verifies)

## 5. Patch decision tree

```
Run cuda5 import test → does class resolve?
├── Yes
│   └── Run integration test with 32 local experts × 8 active dispatch
│       ├── PASS  → no patch needed; Story 3 becomes a thin coercion shim
│       └── FAIL  → branch on traceback
│           ├── int64 dispatch → write int32 cast in shim (Story 3)
│           ├── per-head shape → write flatten + rescale shim (Story 3)
│           ├── sm_90 assert in compiled GEMM → block on Story 4 INT4 backport
│           └── shape '[…]' invalid → o_groups path; Story 6 isolated lane
└── No
    └── Block on CC2 ABI rebuild — class missing means module compile flag drift
```

## 6. EP=8 alltoall expectations (Diamond gating)

Under TP=2×EP=8 the dispatch traffic is **all-to-all** across 8 nodes
(100 GbE RoCEv2). For each token routed to a remote expert:

- **Send:** `H * dtype_bytes` (bf16 → 14336 B/token to one peer)
- **Receive:** `H * dtype_bytes` after the expert runs

At decode (`T=1`, k=8 experts, balanced 1 local + 7 remote per rank):

```
Per-token alltoall = 7 * 2 * 14336 B = 200 KiB
At 50 Gbps = ~6.25 GB/s = 32 µs per decode step
```

Decode budget under Diamond conc=1 (75 t/s = 13.3 ms/token total):
**32 µs alltoall is 0.24% of budget — fabric is not the bottleneck.**
Marlin GEMM + KV reads dominate. This validates EP=8 over TP=16
all-reduce (which would saturate the same fabric every step).

## 7. Hand-off to Story 3

Story 3 (`csrc_patches/moe_dispatch_dsa_sm86.cu`) consumes this audit
and implements:

1. Lightning Indexer output coercion (int32 + softmax-normalize + flatten if per-head).
2. Pre-PREPARE shape adapter for the 256E / 8A layout.
3. `expert_map` builder that maps global expert id → 32-slot local layout.
4. Hook point: `MoEPrepareAndFinalizeNoDPEPModular.prepare` monkey-patch
   path for fast-iteration before sourcing into vLLM proper.

Story 9 (integration test) validates this end-to-end against the
1500-prompt set under TP=2×EP=8.

## 8. Closure criteria

Story 2 closes when:

- [x] Audit document committed (this file).
- [ ] CC1 kernel inventory cross-references this audit with file/line evidence.
- [ ] Symbol-level checklist (§4) has a CC2 rebuild dependency tracked.

Track A continues at Story 3 immediately — does not wait on either of
the unchecked items above.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort, agent: CC5]
