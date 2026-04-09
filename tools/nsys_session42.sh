#!/bin/bash
# Run nsys profiling for BF16, FP8, FP8+stash on an idle GPU.
# Usage: bash tools/nsys_session42.sh
set -e

VENV="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate"
PROJECT="/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe"
OUTDIR="${PROJECT}/reports/nsys_s42"
GPU=${CUDA_VISIBLE_DEVICES:-2}

source "${VENV}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH}"
mkdir -p "${OUTDIR}"

for MODE in fp8 fp8_stash; do
    echo "=== Profiling ${MODE} ==="
    CUDA_VISIBLE_DEVICES=${GPU} nsys profile \
        --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --force-overwrite=true \
        -o "${OUTDIR}/nsys_s42_${MODE}" \
        python tools/nsys_benchmark.py --mode "${MODE}" --gpu 0 --warmup 5 --iters 10
    echo ""
done

echo "=== Extracting GPU Projection from SQLite ==="
for MODE in fp8 fp8_stash; do
    DB="${OUTDIR}/nsys_s42_${MODE}.sqlite"
    if [ -f "${DB}" ]; then
        echo "  ${MODE}:"
        python -c "
import sqlite3, statistics
db = sqlite3.connect('${DB}')
rows = db.execute('''
    SELECT start, end, demangledName
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    ORDER BY start
''').fetchall()
# Sum kernel durations per NVTX range (forward_N / backward_N)
fwd_sums, bwd_sums = [], []
for r in rows:
    dur_us = (r[1] - r[0]) / 1000  # ns to us
total_us = sum((r[1]-r[0])/1000 for r in rows)
n_iters = 10
per_iter_us = total_us / n_iters
print(f'    Total GPU kernel time: {total_us:.0f} us ({n_iters} iters)')
print(f'    Per-iter GPU Projection: {per_iter_us:.0f} us')
db.close()
"
    fi
done
echo "=== Done ==="
