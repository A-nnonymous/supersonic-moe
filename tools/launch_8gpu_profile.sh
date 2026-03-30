#!/bin/bash
set -e
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
NSYS=/opt/nvidia/nsight-systems-cli/2025.1.1/bin/nsys

# GPU 0: nsys capture
echo "[launcher] GPU 0 → nsys"
CUDA_VISIBLE_DEVICES=0 $NSYS profile \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o /tmp/sonic_async_v3 --force-overwrite true \
  -t cuda,nvtx --gpu-metrics-device=0 \
  python /tmp/gpu_async_profile.py > /tmp/gpu_async_log_0.txt 2>&1 &
pids[0]=$!

# GPU 1-7: pure benchmark
for g in 1 2 3 4 5 6 7; do
  echo "[launcher] GPU $g → bench"
  CUDA_VISIBLE_DEVICES=$g python /tmp/gpu_async_profile.py \
    > /tmp/gpu_async_log_${g}.txt 2>&1 &
  pids[$g]=$!
done

echo "[launcher] waiting..."
fail=0
for g in 0 1 2 3 4 5 6 7; do
  wait ${pids[$g]} && echo "GPU $g OK" || { echo "GPU $g FAIL"; fail=$((fail+1)); }
done
echo "All done. Failures: $fail/8"

# ── aggregate ──
python3 << 'PYEOF'
import json, os, statistics

R = []
for g in range(8):
    p = f"/tmp/gpu_async_result_{g}.json"
    if os.path.exists(p):
        R.append(json.load(open(p)))
n = len(R)
if not n: print("No results!"); exit(1)

med = statistics.median
W, IT = R[0]["warmup"], R[0]["iters"]

print(f"\n{'='*92}")
print(f"AGGREGATE {n} GPUs | {W}w {IT}i | async CUDA-event (no artificial sync)")
print(f"T=4096 H=4096 I=1024 E=128 K=8 tpe=256")
print(f"{'='*92}")

def g(key, sub):
    return [r[key][sub]["med"] for r in R]

hdr = f"{'':18} {'BF16 med(ms)':>14} {'FP8 med(ms)':>14} {'Speedup':>10}"
print(f"\n{hdr}\n{'-'*58}")
for lab, bk, fk in [("Forward","fwd","fwd"),("Backward","bwd","bwd"),("Total","total","total")]:
    bv, fv = med(g("bf16",bk)), med(g("fp8",fk))
    print(f"{lab:<18} {bv:>14.3f} {fv:>14.3f} {bv/fv:>9.2f}x")

bb = med(g("bf16","bubble")); fb = med(g("fp8","bubble"))
bt = med(g("bf16","total")); ft = med(g("fp8","total"))
print(f"{'Bubble':<18} {bb:>14.3f} {fb:>14.3f}")
print(f"{'Bubble %':<18} {bb/bt*100:>13.1f}% {fb/ft*100:>13.1f}%")

print(f"\n{'GPU':>5} {'BF16f':>8} {'FP8f':>8} {'BF16b':>8} {'FP8b':>8} "
      f"{'BF16t':>8} {'FP8t':>8} {'Spf':>6} {'Spb':>6} {'Spt':>6}")
for r in R:
    print(f"{'G'+str(r['gpu']):>5}"
          f" {r['bf16']['fwd']['med']:>8.3f} {r['fp8']['fwd']['med']:>8.3f}"
          f" {r['bf16']['bwd']['med']:>8.3f} {r['fp8']['bwd']['med']:>8.3f}"
          f" {r['bf16']['total']['med']:>8.3f} {r['fp8']['total']['med']:>8.3f}"
          f" {r['speedup']['fwd']:>5.2f}x {r['speedup']['bwd']:>5.2f}x {r['speedup']['total']:>5.2f}x")

print(f"\n{'Metric':<18} {'Mean':>10} {'Min':>10} {'Max':>10} {'Thr':>8} {'OK':>4}")
print(f"{'-'*62}")
for k in R[0]["precision"]:
    vs = [r["precision"][k] for r in R]
    mn, mx, av = min(vs), max(vs), statistics.mean(vs)
    if 'rmse' in k:
        ok = "✓" if mx<0.10 else "✗"
        print(f"{k:<18} {av*100:>9.2f}% {mn*100:>9.2f}% {mx*100:>9.2f}% {'<10%':>8} {ok:>4}")
    else:
        ok = "✓" if mn>0.99 else "✗"
        print(f"{k:<18} {av:>10.5f} {mn:>10.5f} {mx:>10.5f} {'>0.99':>8} {ok:>4}")
PYEOF
