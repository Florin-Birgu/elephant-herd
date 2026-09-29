#!/usr/bin/env bash
# DeepSeek on every test at the final settings, one model reading everything and the
# herd side by side, with per-call logging. Wafer host: fast and cheapest allowed.
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
M=deepseek/deepseek-v4.1-flash; P=wafer; S=deepseek-deepseek-v4-1-flash
run() {  # task length
  local out="runs/final2-$1-$2-$S"
  python3 -m herd.ruler --data "corpora/ruler/$1_$2.jsonl" --n 7 --limit 10 \
    --arms single herd --model "$M" --provider "$P" --prompt report_not_judge \
    --max-tokens 60000 --out "$out.jsonl" > "$out.log" 2>&1
}
for t in niah_single_2 niah_multivalue vt; do run "$t" 128k & done
run vt 256k &
run vt 512k &
wait
echo DS_DONE
