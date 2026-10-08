#!/bin/bash
# Run one handoff stage in the vLLM venv, log it, then sync results and logs to S3.
# Usage: hx_run.sh <logname> <script.py> [args...]
. ~/.moefy_env; cd /mnt/nvme/moefy && . .venv-vllm/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
name=$1; shift
mkdir -p /mnt/nvme/moefy/logs /mnt/nvme/moefy/results/handoff
timeout 14400 python "$@" >> /mnt/nvme/moefy/logs/hx_$name.log 2>&1 && echo "HX_OK $name" >> /mnt/nvme/moefy/logs/hx_$name.log || echo "HX_FAIL $name" >> /mnt/nvme/moefy/logs/hx_$name.log
aws s3 sync /mnt/nvme/moefy/results/handoff s3://$MOEFY_BUCKET/handoff-results/ --only-show-errors
aws s3 sync /mnt/nvme/moefy/logs s3://$MOEFY_BUCKET/handoff-results/logs/ --only-show-errors --exclude "*" --include "hx_*"
echo "HX_DONE $name" >> /mnt/nvme/moefy/logs/hx_$name.log
