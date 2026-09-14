#!/bin/bash
# Fetch the normalised benchmark test files (LLMOPT release) from GitHub.
set -euo pipefail
cd "$(dirname "$0")"
declare -A MAP=( [nl4opt]=nl4opt_test [mamo_easy]=mamo_easy_test [mamo_complex]=mamo_complex_test
                 [complexor]=complexor [industryor]=industryor [optibench]=optibench [optmath]=optmath_bench )
missing=0
for k in "${!MAP[@]}"; do [ -f "benchmarks/$k.jsonl" ] || missing=1; done
if [ "$missing" = 1 ]; then
  rm -rf _llmopt
  git clone -q --depth 1 https://github.com/caigaojiang/LLMOPT.git _llmopt
  mkdir -p benchmarks
  for k in "${!MAP[@]}"; do cp "_llmopt/data/testset/${MAP[$k]}.jsonl" "benchmarks/$k.jsonl"; done
  rm -rf _llmopt
fi
wc -l benchmarks/*.jsonl
