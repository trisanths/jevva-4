#!/bin/bash
# 26B one-graph Jev path: int4 experts check + bench at 280, 140, 70; then accuracy for FP8 and int4 at 280.
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 3600 env "$@" >> /mnt/nvme/moefy/logs/f26b.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/f26b.log; }
run JEV_EXPERTS=int4 python jev_fast26.py check 280
run JEV_EXPERTS=int4 python jev_fast26.py bench 140
run JEV_EXPERTS=int4 python jev_fast26.py bench 70
run JEV_EXPERTS=fp8 python jev_fast26.py bench 140
run JEV_EXPERTS=int4 python jev_fast26.py acc 280
run JEV_EXPERTS=fp8 python jev_fast26.py acc 280
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo F26_DONE >> /mnt/nvme/moefy/logs/f26b.log
