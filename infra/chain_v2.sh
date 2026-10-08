#!/bin/bash
# Where do the 26B's ~107 ms go on this GPU: text-only Jev request in vLLM, and our compiled vision tower alone.
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q V1_DONE logs/v1.log 2>/dev/null; do sleep 30; done
cd code/src
(. ../../.venv-vllm/bin/activate && timeout 2400 python jev_vllm.py lat_text NeoChen1024/gemma-4-26B-A4B-it-qat-W4A16 280 60 > /mnt/nvme/moefy/logs/v2.log 2>&1)
(. ../../.venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && timeout 2400 python tower26.py 280,140,70 >> /mnt/nvme/moefy/logs/v2.log 2>&1)
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo V2_DONE >> /mnt/nvme/moefy/logs/v2.log
