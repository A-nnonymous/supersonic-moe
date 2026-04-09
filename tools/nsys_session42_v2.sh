#!/bin/bash
set -e
PROJECT=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
OUTDIR=${PROJECT}/reports/nsys_s42
VENV=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
PYTHON=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python
GPU=${CUDA_VISIBLE_DEVICES:-2}

source "${VENV}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH}"
mkdir -p "${OUTDIR}"

# Export existing FP8 nsys-rep to sqlite
if [ -f "${OUTDIR}/nsys_s42_fp8.nsys-rep" ] && [ ! -f "${OUTDIR}/nsys_s42_fp8.sqlite" ]; then
    echo "Exporting FP8 nsys-rep to sqlite..."
    nsys export --type sqlite --force-overwrite true -o "${OUTDIR}/nsys_s42_fp8.sqlite" "${OUTDIR}/nsys_s42_fp8.nsys-rep"
fi

# Profile FP8+stash if not already done
if [ ! -f "${OUTDIR}/nsys_s42_fp8_stash.nsys-rep" ]; then
    echo "=== Profiling FP8+stash ==="
    CUDA_VISIBLE_DEVICES=${GPU} nsys profile \
        --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --cuda-event-trace=false \
        --force-overwrite=true \
        -o "${OUTDIR}/nsys_s42_fp8_stash" \
        ${PYTHON} tools/nsys_benchmark.py --mode fp8_stash --gpu 0 --warmup 5 --iters 10
fi

# Export fp8_stash to sqlite
if [ -f "${OUTDIR}/nsys_s42_fp8_stash.nsys-rep" ] && [ ! -f "${OUTDIR}/nsys_s42_fp8_stash.sqlite" ]; then
    echo "Exporting FP8+stash nsys-rep to sqlite..."
    nsys export --type sqlite --force-overwrite true -o "${OUTDIR}/nsys_s42_fp8_stash.sqlite" "${OUTDIR}/nsys_s42_fp8_stash.nsys-rep"
fi

# Extract GPU kernel time
echo ""
echo "=== GPU Projection Summary ==="
${PYTHON} - <<'PYEOF'
import sqlite3, os
outdir = os.environ.get("OUTDIR", "reports/nsys_s42")
for mode in ["fp8", "fp8_stash"]:
    db_path = f"{outdir}/nsys_s42_{mode}.sqlite"
    if not os.path.exists(db_path):
        print(f"  {mode}: sqlite not found at {db_path}")
        continue
    db = sqlite3.connect(db_path)
    rows = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    total_ns = sum(r[1] - r[0] for r in rows)
    n_iters = 10
    per_iter_us = total_ns / 1000 / n_iters
    print(f"  {mode:12s}: {len(rows):5d} kernels, total={total_ns/1000:.0f} us, per_iter={per_iter_us:.0f} us")
    db.close()
PYEOF

echo "=== Done ==="
