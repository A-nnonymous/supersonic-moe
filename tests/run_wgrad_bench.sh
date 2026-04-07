#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python tests/bench_fp8_wgrad_fast.py
