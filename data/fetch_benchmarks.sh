#!/bin/bash
# Fetch the normalised benchmark test files (LLMOPT release) from GitHub.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f benchmarks/nl4opt.jsonl ] || [ ! -f benchmarks/mamo_easy.jsonl ]; then
  rm -rf _llmopt
  git clone -q --depth 1 https://github.com/caigaojiang/LLMOPT.git _llmopt
  mkdir -p benchmarks
  cp _llmopt/data/testset/nl4opt_test.jsonl benchmarks/nl4opt.jsonl
  cp _llmopt/data/testset/mamo_easy_test.jsonl benchmarks/mamo_easy.jsonl
  rm -rf _llmopt
fi
wc -l benchmarks/*.jsonl
