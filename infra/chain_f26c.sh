#!/bin/bash
# End-to-end 26B Jev latency (uint8 frame + question in, probabilities out) with Marlin 4-bit experts.
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q F26B_DONE logs/f26c.log 2>/dev/null; do sleep 30; done
. .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for b in 280 140 70; do timeout 2400 env JEV_EXPERTS=marlin python jev_fast26.py e2e $b >> /mnt/nvme/moefy/logs/f26d.log 2>&1 || echo "TIMEOUT_OR_FAIL $b" >> /mnt/nvme/moefy/logs/f26d.log; done
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo F26C_DONE >> /mnt/nvme/moefy/logs/f26d.log
