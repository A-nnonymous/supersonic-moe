#!/bin/bash
set -e
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
export USE_QUACK_GEMM=1
export SONIC_MOE_FP8_MODE=perf

echo "=== MoE core tests ==="
python -m pytest tests/moe_test.py tests/moe_blackwell_test.py -x -q --tb=short

echo ""
echo "=== FP8 protocol tests ==="
python -m pytest tests/fp8_protocol_test.py -x -q --tb=short

echo ""
echo "=== FP8 contract tests ==="
python -m pytest tests/fp8_large_project_contract_test.py -x -q --tb=short

echo ""
echo "=== Blockscaled FP8 varlen tests ==="
python -m pytest tests/test_blockscaled_fp8_varlen.py -x -q --tb=short

echo ""
echo "=== ALL DONE ==="
