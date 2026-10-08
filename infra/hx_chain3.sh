#!/bin/bash
# Full A2 (one epoch, eval after), the C1 relay control, then the latency bench of the trained A2 map.
. ~/.moefy_env
cd /mnt/nvme/moefy
./hx_run.sh a2_full hx_train.py A2 1
./hx_run.sh c1 hx_c1.py C1 1
./hx_run.sh bench_a2 hx_bench.py /mnt/nvme/moefy/hx/b70/A2.pt A2
echo CHAIN3_DONE >> logs/hx_chain3.log
aws s3 cp logs/hx_chain3.log s3://$MOEFY_BUCKET/handoff-results/logs/hx_chain3.log --only-show-errors
