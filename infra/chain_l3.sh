#!/bin/bash
# L40S latency benches: per-tensor FP8 (rowwise FP8 hangs under CUDA graph replay on sm89), then compiled bf16,
# then compiled FP8. Each run has a 20-minute timeout so one hang cannot stall the queue.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 1200 python fast_bench_e.py "$@" >> /mnt/nvme/moefy/logs/bench_l3.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/bench_l3.log; }
JEV_FP8=tensor run google/gemma-4-E2B-it 70,140,280
JEV_FP8=tensor run google/gemma-4-E4B-it 70,140,280
JEV_COMPILE=default run google/gemma-4-E2B-it 70,140,280
JEV_COMPILE=default run google/gemma-4-E4B-it 70,140,280
JEV_FP8=tensor JEV_COMPILE=default run google/gemma-4-E2B-it 70,140,280
JEV_FP8=tensor JEV_COMPILE=default run google/gemma-4-E4B-it 70,140,280
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L3_DONE >> /mnt/nvme/moefy/logs/bench_l3.log
