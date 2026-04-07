#!/bin/bash
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
export USE_QUACK_GEMM=1
export SONIC_MOE_FP8_MODE=perf
python -m pytest tests/moe_test.py tests/moe_blackwell_test.py -x -q --tb=short
echo "EXIT_CODE=$?"
