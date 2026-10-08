#!/bin/bash
# Push the Jev-tuned E4B toward 10 ms: end-to-end latency for FP8 vs FP8+int4-tail, default vs max-autotune compile.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/mnt/nvme/moefy/adapters/e4b-jev
run() { timeout 2400 python e2e_bench.py "$@" >> /mnt/nvme/moefy/logs/l9.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/l9.log; }
JEV_FP8=tensor run google/gemma-4-E4B-it 70 $A
JEV_FP8=mixed run google/gemma-4-E4B-it 70 $A
JEV_FP8=mixed JEV_COMPILE=max-autotune-no-cudagraphs run google/gemma-4-E4B-it 70 $A
JEV_FP8=mixed JEV_COMPILE=max-autotune-no-cudagraphs run google/gemma-4-E2B-it 70 /mnt/nvme/moefy/adapters/e2b-jev
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L9_DONE >> /mnt/nvme/moefy/logs/l9.log
