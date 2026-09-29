#!/usr/bin/env bash
# Old settings vs new settings, same models, same tasks, same samples.
#
#   old: --prompt answer --max-tokens 2000   (what every result before the fix used)
#   new: --prompt plain  --max-tokens 20000
#
# The new-settings runs for "one fact" and "chains" already exist at n=10 on the same
# data files (runs/final-*), so only the old settings and the new "several facts" run
# (previously n=5) are run here. Sonnet 5 is left out: ~$0.36 a sample.
#
#   scripts/ab_rerun.sh
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a

N=10
rm -f corpora/ruler/niah_multivalue_128k.jsonl
for task in niah_single_2 niah_multivalue vt; do
  python3 -m herd.ruler_data "$task" --length 128k --limit "$N" >/dev/null
done

MODELS=(
  "z-ai/glm-5.3-flash|z-ai/fp8"
  "deepseek/deepseek-v4.1-flash|deepinfra/fp8"
  "google/gemini-3.7-flash|google-ai-studio"
  "google/gemini-2.5-flash-lite|google-ai-studio"
  "qwen/qwen3.7-flash|"
  "xiaomi/mimo-v2.5|xiaomi/fp8"
)

run() {  # model provider setting task prompt cap
  local slug; slug=$(echo "$1" | tr '/.' '--')
  local out="runs/ab-$3-$4-$slug"
  python3 -m herd.ruler --data "corpora/ruler/$4_128k.jsonl" --n 7 --limit "$N" \
    --arms herd --model "$1" ${2:+--provider "$2"} --prompt "$5" --max-tokens "$6" \
    --out "$out.jsonl" > "$out.log" 2>&1
}

run_model() {
  local model="${1%%|*}" provider="${1##*|}"
  for task in niah_single_2 niah_multivalue vt; do
    run "$model" "$provider" old "$task" answer 2000
  done
  run "$model" "$provider" new niah_multivalue plain 20000
  echo "done: $model"
}

for entry in "${MODELS[@]}"; do
  run_model "$entry" &
done
wait
echo "AB COMPLETE"
