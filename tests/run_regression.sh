#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT
export NNODES=1 PADDLE_TRAINERS_NUM=1

echo "=== Running full regression test suite ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Core regression tests
echo "--- fp8_large_project_contract_test.py (31 tests) ---"
USE_QUACK_GEMM=1 python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- fp8_protocol_test.py (26 tests) ---"
USE_QUACK_GEMM=1 python -m pytest tests/fp8_protocol_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- moe_blackwell_test.py ---"
USE_QUACK_GEMM=1 python -m pytest tests/moe_blackwell_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- moe_test.py ---"
USE_QUACK_GEMM=1 python -m pytest tests/moe_test.py -v --tb=short 2>&1 || true

echo ""
echo "=== Regression complete ==="
