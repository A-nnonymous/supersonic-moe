#!/bin/bash
source /root/paddlejob/share-storage/gpfs/system-public/zhangyichen/baidu/ernie/erniebot/eb_venv/bin/activate
export PYTHONPATH=/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/baidu/ernie/erniebot/third_party/ernie-core/src:/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe:/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack:$PYTHONPATH
export SONIC_MOE_FP8_MODE=perf
export USE_QUACK_GEMM=1
export SONIC_MOE_FP8_ASSUME_ALIGNED=1
export QUACK_CACHE_DIR=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/my_quack_cache
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
export NNODES=1
export PADDLE_TRAINERS_NUM=1
export CUDA_VISIBLE_DEVICES=0
