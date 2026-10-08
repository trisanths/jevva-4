#!/bin/bash
# Accuracy of the Jev-tuned E4B through the compiled fast path: bf16 and mixed FP8/int4.
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q L9_DONE logs/l9.log 2>/dev/null; do sleep 30; done
. .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=/mnt/nvme/moefy/adapters/e4b-jev
timeout 3000 python fast_eval_e.py google/gemma-4-E4B-it 70 $A bf16-c-jev > /mnt/nvme/moefy/logs/l10.log 2>&1
JEV_FP8=mixed timeout 3600 python fast_eval_e.py google/gemma-4-E4B-it 70 $A mixed-c-jev >> /mnt/nvme/moefy/logs/l10.log 2>&1
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L10_DONE >> /mnt/nvme/moefy/logs/l10.log
