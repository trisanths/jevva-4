#!/bin/bash
# FP8 vision tower on the 26B fast path: bench at 280/140 and accuracy at 280 (Marlin 4-bit experts).
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q F26C_DONE logs/f26d.log 2>/dev/null; do sleep 30; done
. .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 3600 env "$@" >> /mnt/nvme/moefy/logs/f26e.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/f26e.log; }
run JEV_EXPERTS=marlin JEV_VISION_FP8=1 python jev_fast26.py bench 280
run JEV_EXPERTS=marlin JEV_VISION_FP8=1 python jev_fast26.py bench 140
run JEV_EXPERTS=marlin JEV_VISION_FP8=1 python jev_fast26.py acc 280
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo F26D_DONE >> /mnt/nvme/moefy/logs/f26e.log
