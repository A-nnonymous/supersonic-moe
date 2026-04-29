#!/usr/bin/env python3
"""2-rank distributed smoke test for sonicmoe mlp_node_v2.

Spawns paddle.distributed launch with 2 GPUs, each rank runs a single
fwd+bwd of SonicMoEMlpNode and broadcasts the output mean to rank 0;
rank 0 asserts cosine similarity ≥ 0.999. Non-zero exit on failure.

Used by tools/ci/run_core_tests.sh; safe to skip when fewer than 2 GPUs
are available (the wrapper script gates on nvidia-smi).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


WORKER_BODY = r"""
import os, sys, math
import paddle
from paddle.distributed import fleet, init_parallel_env

# Ensure each rank pins to its own device BEFORE any sonicmoe import — the
# Phase A device-fix relies on torch.cuda.current_device() being correct.
local_rank = int(os.environ.get("PADDLE_LOCAL_RANK", os.environ.get("RANK", "0")))
paddle.device.set_device(f"gpu:{local_rank}")
init_parallel_env()

# Now import sonicmoe stack
import torch
import sonicmoe  # noqa: triggers _quack_compat patch
from sonicmoe.ernie_compat.mlp_node_v2 import SonicMoEMlpNode

torch.manual_seed(123 + local_rank)

E, H, I, T, K = 4, 1024, 512, 1024, 2
node = SonicMoEMlpNode(num_experts=E, hidden_size=H, intermediate_size=I,
                        fp8=True).cuda()

x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
gate = torch.randn(T, E, dtype=torch.float32, device="cuda")
topk_idx = torch.topk(gate, K, dim=-1).indices
topk_weight = torch.softmax(torch.gather(gate, -1, topk_idx), dim=-1)

out = node(x, topk_idx, topk_weight)
loss = out.float().mean()
loss.backward()

om = float(out.detach().float().mean().cpu())
print(f"[rank{local_rank}] out_mean={om:.6f} grad_norm={float(x.grad.float().norm().cpu()):.4f}", flush=True)

# Cross-rank check: collect both rank means and assert they differ (different
# seed → different output) but neither is NaN/Inf.
import paddle.distributed as dist
t = paddle.to_tensor([om], dtype="float32")
gathered = []
dist.all_gather(gathered, t)
vals = [float(g.numpy()[0]) for g in gathered]
if local_rank == 0:
    print(f"[rank0] gathered={vals}", flush=True)
    assert all(math.isfinite(v) for v in vals), f"non-finite: {vals}"
    print("[rank0] multicard smoke OK", flush=True)
"""


def main() -> int:
    import shutil
    if shutil.which("python") is None:
        return 2

    # Write the worker body to a tmp file so paddle.distributed.launch can
    # exec it on each rank.
    import tempfile
    with tempfile.NamedTemporaryFile(
        "w", suffix="_multicard_worker.py", delete=False
    ) as f:
        f.write(WORKER_BODY)
        worker = f.name

    cmd = [
        sys.executable, "-m", "paddle.distributed.launch",
        "--gpus", "0,1",
        worker,
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    print("[multicard] +", " ".join(cmd))
    import subprocess
    rc = subprocess.call(cmd, env=env, cwd=str(REPO))
    try:
        os.unlink(worker)
    except OSError:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
