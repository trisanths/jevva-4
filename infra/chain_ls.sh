#!/bin/bash
# 26B LM latency against sequence length (Marlin 4-bit experts, compiled graph).
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 3600 env JEV_EXPERTS=marlin python lm_short.py >> /mnt/nvme/moefy/logs/ls.log 2>&1 || echo TIMEOUT_OR_FAIL >> /mnt/nvme/moefy/logs/ls.log
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo LS_DONE >> /mnt/nvme/moefy/logs/ls.log
