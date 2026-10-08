#!/bin/bash
# Jev-tune E2B on the L40S, then score it through the HF path (all budgets) and the compiled fast path (70).
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q L4_DONE logs/l4.log 2>/dev/null; do sleep 20; done
. .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JEV_FP8=tensor timeout 2400 python fast_eval_e.py google/gemma-4-E2B-it 70 - fp8-c > /mnt/nvme/moefy/logs/eval_fp8.log 2>&1
JEV_MID=google/gemma-4-E2B-it JEV_MICRO=4 python train_jev.py 35 35 30000 e2b-jev > /mnt/nvme/moefy/logs/train_e2b.log 2>&1
python jev_base.py google/gemma-4-E2B-it 280,140,70 /mnt/nvme/moefy/adapters/e2b-jev jev > /mnt/nvme/moefy/logs/eval_e2b_jev.log 2>&1
python fast_eval_e.py google/gemma-4-E2B-it 70 /mnt/nvme/moefy/adapters/e2b-jev >> /mnt/nvme/moefy/logs/eval_e2b_jev.log 2>&1
JEV_FP8=tensor python fast_eval_e.py google/gemma-4-E2B-it 70 /mnt/nvme/moefy/adapters/e2b-jev fp8-c-jev >> /mnt/nvme/moefy/logs/eval_e2b_jev.log 2>&1
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
aws s3 sync /mnt/nvme/moefy/adapters s3://$MOEFY_BUCKET/adapters --only-show-errors
echo L5_DONE >> /mnt/nvme/moefy/logs/eval_e2b_jev.log
