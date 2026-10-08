#!/bin/bash
# after chain1: latency bench on the A1 map, then a 30-step A2 smoke test
cd /mnt/nvme/moefy
until grep -q CHAIN1_DONE logs/hx_chain1.log 2>/dev/null; do sleep 20; done
./hx_run.sh bench_a1 hx_bench.py /mnt/nvme/moefy/hx/b70/ridge.pt A1-ridge
HX_MAX_STEPS=30 ./hx_run.sh a2_smoke hx_train.py smoke
echo CHAIN2_DONE >> logs/hx_chain2.log
