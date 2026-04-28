#!/bin/bash
# --ProtoAI-Bakari--
# CC0 directive (b): MMLU full-corpus loader for perplexity_full.py + gen_prompts.py.
# Downloads cais/mmlu dataset from HuggingFace via git clone (LFS-aware) into a
# local directory, exporting MMLU_DIR for downstream tools.
#
# Run:
#   bash download_mmlu.sh                              # default ~/datasets/mmlu
#   MMLU_DIR=/path/to/dest bash download_mmlu.sh
#
# After completion:
#   export MMLU_DIR=~/datasets/mmlu
#   python3 gen_prompts.py               # mmlu_real category populated
#   python3 perplexity_full.py ...       # full per-subject perplexity
#
# Layout produced: $MMLU_DIR/{test,dev,val}/{subject}_test.csv
#   ~14K test questions across 57 subjects.
set -euo pipefail
DEST="${MMLU_DIR:-${HOME}/datasets/mmlu}"

if [[ -d "$DEST/test" ]] && [[ -n "$(ls -A "$DEST/test"/*.csv 2>/dev/null || true)" ]]; then
  count=$(ls "$DEST/test"/*.csv 2>/dev/null | wc -l | tr -d ' ')
  echo "[download_mmlu] $DEST already has $count test CSVs — skipping clone"
  echo "[download_mmlu] re-export: export MMLU_DIR=$DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"

if [[ -d "$DEST/.git" ]]; then
  echo "[download_mmlu] $DEST already a git repo — pulling latest"
  cd "$DEST" && git pull --ff-only
  exit 0
fi

# Try HF git clone (LFS-aware if installed)
URL_PRIMARY="https://huggingface.co/datasets/cais/mmlu"
URL_FALLBACK="https://huggingface.co/datasets/lukaemon/mmlu"

echo "[download_mmlu] cloning $URL_PRIMARY → $DEST"
if git clone --depth=1 "$URL_PRIMARY" "$DEST" 2>&1; then
  echo "[download_mmlu] primary clone succeeded"
elif git clone --depth=1 "$URL_FALLBACK" "$DEST" 2>&1; then
  echo "[download_mmlu] fallback clone succeeded"
else
  echo "[download_mmlu] git clone failed — falling back to per-subject curl" >&2
  mkdir -p "$DEST/test"
  for subj in abstract_algebra anatomy astronomy business_ethics clinical_knowledge \
              college_biology college_chemistry college_computer_science college_mathematics \
              college_medicine college_physics computer_security conceptual_physics \
              econometrics electrical_engineering elementary_mathematics formal_logic \
              global_facts high_school_biology high_school_chemistry high_school_computer_science \
              high_school_european_history high_school_geography high_school_government_and_politics \
              high_school_macroeconomics high_school_mathematics high_school_microeconomics \
              high_school_physics high_school_psychology high_school_statistics high_school_us_history \
              high_school_world_history human_aging human_sexuality international_law \
              jurisprudence logical_fallacies machine_learning management marketing \
              medical_genetics miscellaneous moral_disputes moral_scenarios nutrition \
              philosophy prehistory professional_accounting professional_law professional_medicine \
              professional_psychology public_relations security_studies sociology us_foreign_policy \
              virology world_religions; do
    out="$DEST/test/${subj}_test.csv"
    if [[ -s "$out" ]]; then continue; fi
    /usr/bin/curl -sSL --max-time 30 -o "$out" \
      "https://huggingface.co/datasets/cais/mmlu/resolve/main/test/${subj}_test.csv" || true
  done
fi

count=$(ls "$DEST/test"/*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "[download_mmlu] complete: $count test CSVs at $DEST"
echo "[download_mmlu] export: export MMLU_DIR=$DEST"
