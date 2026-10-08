#!/bin/bash
# Stage split of the quantized (FP8 + int4 tail) models, then accuracy of the tuned models through the fast path.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JEV_FP8=mixed timeout 1800 python prof_stages.py google/gemma-4-E4B-it 70 > /mnt/nvme/moefy/logs/l11.log 2>&1
JEV_FP8=mixed timeout 1800 python prof_stages.py google/gemma-4-E2B-it 70 >> /mnt/nvme/moefy/logs/l11.log 2>&1
echo STAGES_DONE >> /mnt/nvme/moefy/logs/l11.log
A4=/mnt/nvme/moefy/adapters/e4b-jev; A2=/mnt/nvme/moefy/adapters/e2b-jev
timeout 3000 python fast_eval_e.py google/gemma-4-E4B-it 70 $A4 bf16-c-jev >> /mnt/nvme/moefy/logs/l11.log 2>&1
JEV_FP8=mixed timeout 3600 python fast_eval_e.py google/gemma-4-E4B-it 70 $A4 mixed-c-jev >> /mnt/nvme/moefy/logs/l11.log 2>&1
JEV_FP8=mixed timeout 3600 python fast_eval_e.py google/gemma-4-E2B-it 70 $A2 mixed-c-jev >> /mnt/nvme/moefy/logs/l11.log 2>&1
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L11_DONE >> /mnt/nvme/moefy/logs/l11.log
