#!/bin/bash
# ridge (reuse logged selection; final maps for [4,9,14] and [2,6,10,14]) -> eval precompute -> A1 eval -> train precompute
cd /mnt/nvme/moefy
HX_SKIP_SEL=1 HX_SETS="4,9,14;2,6,10,14" ./hx_run.sh ridge2 hx_ridge.py
./hx_run.sh prep_eval hx_prep.py eval
./hx_run.sh a1_eval hx_eval.py A1-ridge /mnt/nvme/moefy/hx/b70/ridge.pt
./hx_run.sh prep_train hx_prep.py train
echo CHAIN1_DONE >> /mnt/nvme/moefy/logs/hx_chain1.log
