#!/bin/bash
# After the L40S benchmark queue: per-stage compiled timings, then accuracy of the deployable compiled path.
. ~/.moefy_env; cd /mnt/nvme/moefy
until grep -q L3_DONE logs/bench_l3.log 2>/dev/null; do sleep 20; done
. .venv/bin/activate && cd code/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 2400 python "$@" >> /mnt/nvme/moefy/logs/l4.log 2>&1 || echo "TIMEOUT_OR_FAIL $*" >> /mnt/nvme/moefy/logs/l4.log; }
run prof_stages.py google/gemma-4-E4B-it 70
run prof_stages.py google/gemma-4-E2B-it 70
run fast_eval_e.py google/gemma-4-E2B-it 70
run fast_eval_e.py google/gemma-4-E4B-it 70
aws s3 sync /mnt/nvme/moefy/results s3://$MOEFY_BUCKET/jev-results-l --only-show-errors
echo L4_DONE >> /mnt/nvme/moefy/logs/l4.log
