#!/bin/bash
# 26B W4A16 in vLLM with the compiled, graph-captured vision encoder.
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q V2_DONE logs/v2.log 2>/dev/null; do sleep 30; done
. .venv-vllm/bin/activate && cd code/src
W4=NeoChen1024/gemma-4-26B-A4B-it-qat-W4A16
run() { JEV_ENC=1 timeout 3600 python jev_vllm.py "$@" >> /mnt/nvme/moefy/logs/v3.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/v3.log; }
run lat $W4 280 60
run lat $W4 140 60
run lat $W4 70 60
run acc $W4 280
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo V3_DONE >> /mnt/nvme/moefy/logs/v3.log
