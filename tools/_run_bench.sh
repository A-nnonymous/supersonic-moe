#!/bin/bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
export CUDA_VISIBLE_DEVICES=0
export USE_QUACK_GEMM=1
export NNODES=1
export PADDLE_TRAINERS_NUM=1
export PYTHONUNBUFFERED=1
unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
unset SONIC_MOE_FP8_MODE SONIC_MOE_FP8_ASSUME_ALIGNED SONIC_MOE_FP8_FUSED_SWIGLU_QUANT SONIC_MOE_FP8_SAVE_Z_FP8 SONIC_MOE_FP8_FUSED_GATED SONIC_MOE_FP8_WGRAD
python -u tools/_run_benchmark.py > /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/bench_result.txt 2>&1
echo "EXIT_CODE=$?" >> /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/output/bench_result.txt
