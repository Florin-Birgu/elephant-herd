#!/usr/bin/env bash
# RAG baselines for the handbook test: embedding search and keyword search (BM25),
# top 10 chunks, same answering model, questions and judge as the herd run.
set -u
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
python3 -m herd.handbook_rag --retriever embed > runs/handbook-rag-embed.log 2>&1 &
python3 -m herd.handbook_rag --retriever bm25 > runs/handbook-rag-bm25.log 2>&1 &
python3 -m herd.handbook_rag --retriever embed --k 30 --out runs/handbook-rag-embed-k30-deepseek-deepseek-v4-1-flash.jsonl > runs/handbook-rag-embed-k30.log 2>&1 &
wait
echo RAG_DONE
