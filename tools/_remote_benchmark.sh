#!/bin/bash
set -e
export VSCODE_SHELL_INTEGRATION=0
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

echo "=== 1. PRECISION (3 seeds, GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python tools/precision_audit.py --gpu 0 --seeds 42,123,777

echo ""
echo "=== 2. MEMORY: Official BF16 (GPU 1) ==="
CUDA_VISIBLE_DEVICES=1 /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/python /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/tools/_mem_bf16_official.py

echo ""
echo "=== 3. MEMORY: FP8 Frontier (GPU 2) ==="
CUDA_VISIBLE_DEVICES=2 /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/tools/_mem_fp8_optimized.py

echo ""
echo "=== 4. NSYS: FP8 (GPU 3) ==="
dpkg -i /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/NsightSystems-linux-cli-public-2025.1.1.131-3554042.deb > /dev/null 2>&1 || true
rm -f /tmp/sonic_remote_fp8.nsys-rep /tmp/sonic_remote_fp8.sqlite
CUDA_VISIBLE_DEVICES=3 nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --sample=none \
  -o /tmp/sonic_remote_fp8 \
  /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/tools/_nsys_fp8_shadow.py 2>&1 | tail -3
nsys export --type=sqlite --output=/tmp/sonic_remote_fp8.sqlite /tmp/sonic_remote_fp8.nsys-rep 2>&1 | tail -1

echo ""
echo "=== 5. NSYS ANALYSIS ==="
/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/sonic_remote_fp8.sqlite')
c = conn.cursor()
c.execute('''SELECT s.value, count(*), sum(k.end-k.start)/1000.0
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
    GROUP BY s.value ORDER BY sum(k.end-k.start) DESC LIMIT 15''')
rows = list(c.fetchall())
total = sum(r[2] for r in rows)
per = total / 10
print(f'FP8+wgrad total: {per:.1f} us/iter')
for name, cnt, tot in rows:
    print(f'  {tot/10:7.1f}us ({tot/total*100:4.1f}%) [{cnt:3d}x] {name[:65]}')
conn.close()
"

echo ""
echo "=== 6. NSYS: BF16 (GPU 3) ==="
rm -f /tmp/sonic_remote_bf16.nsys-rep /tmp/sonic_remote_bf16.sqlite
CUDA_VISIBLE_DEVICES=3 nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --sample=none \
  -o /tmp/sonic_remote_bf16 \
  /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/official_bf16/bin/python /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/tools/_nsys_bf16_official.py 2>&1 | tail -3
nsys export --type=sqlite --output=/tmp/sonic_remote_bf16.sqlite /tmp/sonic_remote_bf16.nsys-rep 2>&1 | tail -1

echo ""
echo "=== 7. BF16 NSYS ANALYSIS ==="
/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/sonic_remote_bf16.sqlite')
c = conn.cursor()
c.execute('''SELECT s.value, count(*), sum(k.end-k.start)/1000.0
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
    GROUP BY s.value ORDER BY sum(k.end-k.start) DESC LIMIT 12''')
rows = list(c.fetchall())
total = sum(r[2] for r in rows)
per = total / 10
print(f'BF16 total: {per:.1f} us/iter')
for name, cnt, tot in rows:
    print(f'  {tot/10:7.1f}us ({tot/total*100:4.1f}%) [{cnt:3d}x] {name[:65]}')
conn.close()
"
echo ""
echo "=== ALL COMPLETE ==="
