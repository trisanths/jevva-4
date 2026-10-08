#!/bin/bash
# Ablation of the Jev-specific savings (after the end-to-end queue).
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q L7_DONE logs/l7.log 2>/dev/null; do sleep 30; done
. .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
timeout 2400 python ablate_e.py google/gemma-4-E2B-it 70 > /mnt/nvme/moefy/logs/l8.log 2>&1
timeout 2400 python ablate_e.py google/gemma-4-E4B-it 70 >> /mnt/nvme/moefy/logs/l8.log 2>&1
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L8_DONE >> /mnt/nvme/moefy/logs/l8.log
