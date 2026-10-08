#!/bin/bash
# Best config (Marlin 4-bit experts + FP8 vision tower): end-to-end at 280/140/70 and accuracy at 140/70.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 3600 env "$@" >> /mnt/nvme/moefy/logs/f26f.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/f26f.log; }
for b in 280 140 70; do run JEV_EXPERTS=marlin JEV_VISION_FP8=1 python jev_fast26.py e2e $b; done
run JEV_EXPERTS=marlin JEV_VISION_FP8=1 python jev_fast26.py acc 140
run JEV_EXPERTS=marlin JEV_VISION_FP8=1 python jev_fast26.py acc 70
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
aws s3 sync /mnt/nvme/moefy/logs s3://$MOEFY_BUCKET/jev-logs-l40s-2 --only-show-errors
echo F26E_DONE >> /mnt/nvme/moefy/logs/f26f.log
