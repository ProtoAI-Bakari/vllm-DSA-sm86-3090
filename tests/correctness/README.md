# tests/correctness — CC6 Gate

The 1500-prompt fixed-seed correctness gate that runs after every CC3/CC4/CC5
integration into the rebuilt `vllm/_C.abi3.so`. CC0 reverts any merge that fails it.

**Gate criteria** (LEAD corpus PART2 §8.4):

| Metric                   | Threshold | Notes                                                  |
|--------------------------|-----------|--------------------------------------------------------|
| top-1 text match         | ≥ 98 %    | deterministic temp=0 → exact match expected            |
| logit cosine             | ≥ 0.97    | requires `logprobs` on both baseline + under-test      |
| MMLU perplexity drop     | ≤ 2 %     | requires `mmlu_real` category + logprobs both sides    |
| word Jaccard (fallback)  | ≥ 0.95    | used when logprobs missing                             |

## Files

| File                  | Purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| `gen_prompts.py`      | Deterministic seed=42 generator for `prompts.jsonl` (1500 lines)        |
| `prompts.jsonl`       | Generated; 9 categories, 1500 total                                     |
| `capture_one.py`      | Single-prompt SSE-streaming POST; writes line to `*.jsonl` w/ file lock |
| `baseline_capture.sh` | Resumable orchestrator; xargs -P CONC over prompts.jsonl                |
| `verify_gate.py`      | Compare baseline + under-test JSONL → markdown report + bridge post     |

## Workflow

### 1. Generate prompts (one time, deterministic)

```bash
python3 gen_prompts.py                          # → prompts.jsonl, seed=42
MMLU_DIR=~/datasets/mmlu python3 gen_prompts.py # use real MMLU corpus for mmlu_real category
```

### 2. Capture baseline (current production endpoint)

```bash
# Default cuda1:8000 vLLM PP (target endpoint per CC0 dispatch — PP fleet on Mac mac4:8000)
BASELINE_ENDPOINT=http://10.255.255.4:8000 \
BASELINE_OUT=baseline.jsonl \
BASELINE_CONC=5 \
BASELINE_LOGPROBS=20 \
bash baseline_capture.sh
```

For MLX-LM servers that reject `logprobs` / `seed` params, use:

```bash
BASELINE_ENDPOINT=http://10.255.255.4:8000 \
BASELINE_OUT=baseline.jsonl \
BASELINE_LOGPROBS=0 \
BASELINE_BACKEND=chat \
bash baseline_capture.sh
```

(Note: `logprobs=0` degrades cosine + perplexity gates to top-1 + Jaccard fallbacks.)

### 3. Capture under-test (after each CC3/CC4/CC5 integration)

```bash
BASELINE_ENDPOINT=http://10.255.255.11:8000 \
BASELINE_OUT=under_test_cc3-rev1.jsonl \
BASELINE_CONC=5 \
bash baseline_capture.sh
```

### 4. Verify

```bash
python3 verify_gate.py \
  --baseline baseline.jsonl \
  --under-test under_test_cc3-rev1.jsonl \
  --integ cc3-rev1 \
  --bridge
# → ~/AGENT/comms/CC6_GATE_RESULTS_cc3-rev1.md
# → exits 0 PASS / 1 FAIL
```

## Categories in prompts.jsonl

| category           | n   | max_tok | purpose                                            |
|--------------------|----:|--------:|----------------------------------------------------|
| factual            | 200 | 16      | short-answer, low-entropy ideal for top-1          |
| mmlu_placeholder   | 300 | 4       | MCQ schema; replaced by `mmlu_real` if MMLU_DIR set|
| reasoning          | 200 | 96      | multi-step word problems                           |
| code               | 200 | 160     | short Python/Bash completions                      |
| summarization      | 150 | 96      | paragraph → 2-sentence summary                     |
| creative           | 150 | 64      | style-constrained 1-2 sentence generation          |
| multilingual       | 100 | 48      | translation                                        |
| repetition         | 100 | 80      | attention-sink adversarial                         |
| edge               | 100 | 64      | empty / 1-tok / special chars / very-long          |
| **total**          |**1500**|         |                                                  |

## Per-integration reports

Reports go to `/Users/z/AGENT/comms/CC6_GATE_RESULTS_<integ>.md` and include
verdict, per-category metrics, and the first 20 token-level disagreements with
baseline vs under-test text side-by-side.
