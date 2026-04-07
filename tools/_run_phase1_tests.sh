#!/bin/bash
set -e
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
export USE_QUACK_GEMM=1
export SONIC_MOE_FP8_MODE=perf

echo "=== Test 1: Regression (epilogue quant OFF) ==="
unset SONIC_MOE_FP8_EPILOGUE_QUANT
python -m pytest tests/moe_test.py tests/moe_blackwell_test.py -x -q --tb=short
echo ""

echo "=== Test 2: Epilogue quant ON ==="
export SONIC_MOE_FP8_EPILOGUE_QUANT=1
python -m pytest tests/moe_test.py tests/moe_blackwell_test.py -x -q --tb=short
echo ""

echo "=== ALL DONE ==="
