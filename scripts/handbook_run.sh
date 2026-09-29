#!/usr/bin/env bash
# Full handbook test: every question and overheard conversation, real herd code,
# product listener prompt, GPT-5.4 judge. Hosts chosen for a cache discount that is
# actually billed (Wafer reports cache hits but charges near full price).
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
python3 -m herd.handbook_bench --model deepseek/deepseek-v4.1-flash --provider deepinfra/fp8 > runs/handbook-deepseek.log 2>&1 &
python3 -m herd.handbook_bench --model z-ai/glm-5.3-flash --provider z-ai/fp8 > runs/handbook-glm.log 2>&1 &
wait
echo HANDBOOK_DONE
