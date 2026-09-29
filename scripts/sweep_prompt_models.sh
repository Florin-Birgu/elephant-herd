#!/usr/bin/env bash
# Chain task, one round, every model against both listener wordings.
#
# The point is the pair: "answer" asks each part for the answer, which no part holds
# when the fact is chained, and "material" asks for the lines instead. Reporting one
# without the other hides that the harness moves the score as much as the model does.
#
# Models run in parallel - they are different hosts and share nothing. The two prompts
# for one model stay sequential, so they do not compete for that host's rate limit or
# thrash its cache.
#
#   scripts/sweep_prompt_models.sh [samples]
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a

SAMPLES="${1:-3}"
DATA=corpora/ruler/vt_128k.jsonl

MODELS=(
  "z-ai/glm-5.3-flash|z-ai/fp8"
  "deepseek/deepseek-v4.1-flash|deepinfra/fp8"
  "google/gemini-3.7-flash|google-ai-studio"
  "qwen/qwen3.7-flash|"
  "anthropic/claude-sonnet-5|anthropic"
)

run_model() {
  local model="${1%%|*}" provider="${1##*|}"
  local slug; slug=$(echo "$model" | tr '/.' '--')
  for prompt in answer material; do
    local out="runs/sweep-${slug}-${prompt}.jsonl"
    if [ -n "$provider" ]; then
      python3 -m herd.ruler --data "$DATA" --n 7 --limit "$SAMPLES" --arms herd \
        --model "$model" --provider "$provider" --prompt "$prompt" --out "$out" \
        > "runs/sweep-${slug}-${prompt}.log" 2>&1
    else
      python3 -m herd.ruler --data "$DATA" --n 7 --limit "$SAMPLES" --arms herd \
        --model "$model" --prompt "$prompt" --out "$out" \
        > "runs/sweep-${slug}-${prompt}.log" 2>&1
    fi
    echo "done: $model $prompt"
  done
}

for entry in "${MODELS[@]}"; do
  run_model "$entry" &
done
wait
echo "SWEEP COMPLETE"
