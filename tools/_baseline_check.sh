#!/bin/bash
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
export USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf
python -m pytest "tests/fp8_protocol_test.py::FP8ProtocolTest::test_blockscaled_downproj_rejects_insufficient_capacity" -x -q --tb=short
