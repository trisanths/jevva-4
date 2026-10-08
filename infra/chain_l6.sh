#!/bin/bash
# FP8 weight-only: latency and accuracy (after the E2B training queue finishes).
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q L5_DONE logs/eval_e2b_jev.log 2>/dev/null; do sleep 30; done
. .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JEV_FP8=wo JEV_COMPILE=default timeout 1500 python fast_bench_e.py google/gemma-4-E2B-it 70,140 > /mnt/nvme/moefy/logs/l6.log 2>&1
JEV_FP8=wo JEV_COMPILE=default timeout 1500 python fast_bench_e.py google/gemma-4-E4B-it 70,140 >> /mnt/nvme/moefy/logs/l6.log 2>&1
JEV_FP8=wo timeout 2400 python fast_eval_e.py google/gemma-4-E2B-it 70 /mnt/nvme/moefy/adapters/e2b-jev fp8wo-c-jev >> /mnt/nvme/moefy/logs/l6.log 2>&1
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L6_DONE >> /mnt/nvme/moefy/logs/l6.log
