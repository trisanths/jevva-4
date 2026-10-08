#!/bin/bash
# vLLM Jev path for the 26B on the L40S: latency (Jev 1 token vs 8 generated tokens), then accuracy.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv-vllm/bin/activate && cd code/src
FP8=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic
W4=NeoChen1024/gemma-4-26B-A4B-it-qat-W4A16
run() { timeout 3600 python jev_vllm.py "$@" >> /mnt/nvme/moefy/logs/v1.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/v1.log; }
run lat $W4 280 60
run lat $FP8 280 60
run acc $W4 280
run acc $FP8 280
run lat $W4 70 60
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo V1_DONE >> /mnt/nvme/moefy/logs/v1.log
