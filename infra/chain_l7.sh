#!/bin/bash
# End-to-end latency (frame + question in, probabilities out) after the FP8 weight-only queue.
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q L6_DONE logs/l6.log 2>/dev/null; do sleep 30; done
. .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/mnt/nvme/moefy/adapters/e2b-jev
run() { timeout 1500 python e2e_bench.py "$@" >> /mnt/nvme/moefy/logs/l7.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/l7.log; }
run google/gemma-4-E2B-it 70 $A
JEV_FP8=tensor run google/gemma-4-E2B-it 70 $A
JEV_FP8=wo run google/gemma-4-E2B-it 70 $A
run google/gemma-4-E2B-it 140 $A
JEV_FP8=tensor run google/gemma-4-E2B-it 140 $A
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L7_DONE >> /mnt/nvme/moefy/logs/l7.log
