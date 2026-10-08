#!/bin/bash
# 26B one-graph Jev path: Marlin 4-bit vs FP8 experts, timings at 280/140/70, then accuracy at 280.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 3600 env "$@" >> /mnt/nvme/moefy/logs/f26c.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/f26c.log; }
run JEV_EXPERTS=marlin python jev_fast26.py check 280
run JEV_EXPERTS=marlin python jev_fast26.py bench 140
run JEV_EXPERTS=marlin python jev_fast26.py bench 70
run JEV_EXPERTS=fp8 python jev_fast26.py bench 140
run JEV_EXPERTS=fp8 python jev_fast26.py bench 70
run JEV_EXPERTS=marlin python jev_fast26.py acc 280
run JEV_EXPERTS=fp8 python jev_fast26.py acc 280
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo F26B_DONE >> /mnt/nvme/moefy/logs/f26c.log
