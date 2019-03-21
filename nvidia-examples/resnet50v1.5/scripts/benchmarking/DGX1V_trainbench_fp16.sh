#!/bin/bash

mkdir -p /tmp/results

python ./scripts/benchmarking/benchmark.py --mode training --precision fp16 --bench-warmup 200 --bench-iterations 500 --ngpus 2 --bs 64 128 256 --baseline /scripts/benchmarking/baselines/RN50_tensorflow_train_fp16.json  --data_dir $1 --results_dir $2