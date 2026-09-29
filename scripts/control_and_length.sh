#!/usr/bin/env bash
# The control and the length test, at the same settings as the model table.
#
#   control: one model reading the whole 128K (single arm) on all three tasks, to set
#            against the herd numbers already in runs/final-* and runs/ab-new-*.
#   length:  chains at 256K and 512K, single and herd, to see whether splitting starts
#            to matter as the text grows.
#
# Two listeners only (GLM-5.3-Flash, DeepSeek-V4.1-Flash): the question is whether
# splitting costs anything for a given model, so the same model on both sides is what
# matters, not how many models. The paper states which models it holds for.
#
#   scripts/control_and_length.sh
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a

N=10
for len in 256k 512k; do python3 -m herd.ruler_data vt --length "$len" --limit "$N" >/dev/null; done

MODELS=(
  "z-ai/glm-5.3-flash|z-ai/fp8"
  "deepseek/deepseek-v4.1-flash|deepinfra/fp8"
)

run() {  # model provider task length arms tag
  local slug; slug=$(echo "$1" | tr '/.' '--')
  local out="runs/$6-$3-$4-$slug"
  python3 -m herd.ruler --data "corpora/ruler/$3_$4.jsonl" --n 7 --limit "$N" \
    --arms $5 --model "$1" --provider "$2" --prompt plain --max-tokens 20000 \
    --out "$out.jsonl" > "$out.log" 2>&1
}

run_model() {
  local model="${1%%|*}" provider="${1##*|}"
  for task in niah_single_2 niah_multivalue vt; do
    run "$model" "$provider" "$task" 128k single control
  done
  for len in 256k 512k; do
    run "$model" "$provider" vt "$len" "single herd" length
  done
  echo "done: $model"
}

for entry in "${MODELS[@]}"; do run_model "$entry" & done
wait
echo "CONTROL AND LENGTH COMPLETE"
