#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
export NNODES=1 PADDLE_TRAINERS_NUM=1
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf python tests/test_fp8_tma_vs_frontier.py
