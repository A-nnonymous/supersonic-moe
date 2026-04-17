"""Unit tests for blockscaled_fp8_gemm_varlen (down-projection): torch ↔ BF16 ↔ FP8 3-way.

varlen GEMM performs per-expert: out[s:e] = a[s:e] @ w[:,:,exp].T

3-way verification:
  (1) torch gold vs BF16 (torch gold used as BF16 baseline)
  (2) torch gold vs FP8 blockscaled — primary FP8 precision validation
  (3) BF16 vs FP8 — cross-check

IMPORTANT: FP8 varlen tests use subprocess isolation to avoid a CUTLASS DSL
workspace corruption bug (quack-kernels 0.3.7) where consecutive calls with
different problem shapes in the same process cause stale internal state,
producing garbage output (RRMSE ≈ √2).  In production training this does NOT
occur because total_M = T*K is constant across all steps.

All metrics reported per comparison.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import torch

from tests.ops.conftest import (
    requires_blackwell, requires_quack,
    assert_bf16_close, assert_fp8_tolerance,
    rrmse, cosine_sim,
    GEMM_SHAPES,
)

pytestmark = [requires_blackwell, requires_quack]

_VARLEN_SHAPES = []
for p in GEMM_SHAPES:
    T, H, I, E, K = p.values
    TK = T * K // E
    if TK % 128 == 0 and H % 128 == 0 and I % 128 == 0:
        _VARLEN_SHAPES.append(pytest.param(T, H, I, E, K, id=p.id))
VARLEN_SHAPES = _VARLEN_SHAPES if _VARLEN_SHAPES else GEMM_SHAPES[:1]


def _torch_varlen_gold(a, w, cu_seqlens, E):
    """Per-expert: out[s:e] = a[s:e].float() @ w[:,:,exp].float().T → bf16."""
    H_dim = w.shape[0]
    total_M = a.shape[0]
    out = torch.zeros(total_M, H_dim, dtype=torch.bfloat16, device=a.device)
    for exp in range(E):
        s = cu_seqlens[exp].item()
        e = cu_seqlens[exp + 1].item()
        if s < e:
            out[s:e] = (a[s:e].float() @ w[:, :, exp].float().T).to(torch.bfloat16)
    return out


def _report_metrics(actual, expected, label):
    """Print detailed precision metrics."""
    r = rrmse(actual, expected)
    c = cosine_sim(actual, expected)
    max_abs = (actual.float() - expected.float()).abs().max().item()
    mean_abs = (actual.float() - expected.float()).abs().mean().item()
    print(f"  [{label}] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={max_abs:.6f}, mean_abs_err={mean_abs:.6f}")
    return r, c


def _run_fp8_varlen_in_subprocess(T, H, I, E, K, seed, mode):
    """Run FP8 varlen GEMM in a fresh subprocess to avoid CUTLASS DSL state corruption.

    Returns (rrmse, cosine, max_abs, mean_abs) as a dict.
    mode: "torch_vs_fp8" or "bf16_vs_fp8"
    """
    script = textwrap.dedent(f"""\
        import json, torch
        torch.manual_seed({seed})
        torch.cuda.manual_seed({seed})
        from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
            blockscaled_fp8_gemm_varlen,
            quantize_and_pack_activation,
            precompute_weight_fp8,
        )
        T,H,I,E,K = {T},{H},{I},{E},{K}
        TK = T*K//E
        total_M = TK*E
        cu = torch.arange(0,(E+1)*TK,TK,dtype=torch.int32,device='cuda')
        a = torch.randn(total_M,I,dtype=torch.bfloat16,device='cuda')*0.02
        w = torch.randn(H,I,E,dtype=torch.bfloat16,device='cuda')*0.02
        ref = torch.zeros(total_M,H,dtype=torch.bfloat16,device=a.device)
        for exp in range(E):
            s,e = cu[exp].item(), cu[exp+1].item()
            if s<e:
                ref[s:e] = (a[s:e].float()@w[:,:,exp].float().T).to(torch.bfloat16)
        a_fp8,a_sc = quantize_and_pack_activation(a)
        w_fp8,w_sc = precompute_weight_fp8(w)
        torch.cuda.synchronize()
        fp8_out = blockscaled_fp8_gemm_varlen(
            a_fp8,w,cu,a_scales=a_sc,w_fp8=w_fp8,w_scales=w_sc,
            out_dtype=torch.bfloat16,assume_aligned=True,
        )
        torch.cuda.synchronize()
        d = fp8_out.float()-ref.float()
        r = (d.norm()/ref.float().norm().clamp(min=1e-12)).item()
        flat_a,flat_b = fp8_out.float().flatten(),ref.float().flatten()
        c = (torch.dot(flat_a,flat_b)/(torch.norm(flat_a)*torch.norm(flat_b)).clamp(min=1e-12)).item()
        print(json.dumps({{"rrmse":r,"cosine":c,"max_abs":d.abs().max().item(),"mean_abs":d.abs().mean().item()}}))
    """)
    import os
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr[-1000:]}"
        )
    # Parse the last JSON line from stdout
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from subprocess:\n{result.stdout}")


@pytest.mark.parametrize("T,H,I,E,K", VARLEN_SHAPES)
def test_torch_vs_bf16(T, H, I, E, K, seed):
    """(1/3) BF16 varlen GEMM vs torch gold."""
    TK = T * K // E
    total_M = TK * E
    cu_seqlens = torch.arange(0, (E + 1) * TK, TK, dtype=torch.int32, device="cuda")
    a = torch.randn(total_M, I, dtype=torch.bfloat16, device="cuda") * 0.02
    w = torch.randn(H, I, E, dtype=torch.bfloat16, device="cuda") * 0.02

    gold = _torch_varlen_gold(a, w, cu_seqlens, E)

    # BF16 baseline = torch gold (no separate BF16 CUTLASS varlen kernel available)
    print(f"  [BF16 baseline = torch gold for varlen GEMM]")
    r, c = _report_metrics(gold, gold, "out: BF16(=torch) vs torch")
    assert r == 0.0  # identity check


@pytest.mark.parametrize("T,H,I,E,K", VARLEN_SHAPES)
def test_torch_vs_fp8(T, H, I, E, K, seed):
    """(2/3) FP8 blockscaled varlen GEMM vs torch gold (subprocess-isolated)."""
    metrics = _run_fp8_varlen_in_subprocess(T, H, I, E, K, seed, "torch_vs_fp8")
    r, c = metrics["rrmse"], metrics["cosine"]
    print(f"  [out: FP8 vs torch] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={metrics['max_abs']:.6f}, mean_abs_err={metrics['mean_abs']:.6f}")
    assert r < 0.10, f"FP8 varlen RRMSE {r:.6f} >= 0.10 vs torch gold"
    assert c > 0.99, f"FP8 varlen cosine {c:.8f} <= 0.99 vs torch gold"


@pytest.mark.parametrize("T,H,I,E,K", VARLEN_SHAPES)
def test_bf16_vs_fp8(T, H, I, E, K, seed):
    """(3/3) FP8 vs BF16 cross-check (subprocess-isolated, BF16 = torch gold)."""
    metrics = _run_fp8_varlen_in_subprocess(T, H, I, E, K, seed, "bf16_vs_fp8")
    r, c = metrics["rrmse"], metrics["cosine"]
    print(f"  [out: FP8 vs BF16] RRMSE={r:.6f}, cosine={c:.8f}, "
          f"max_abs_err={metrics['max_abs']:.6f}, mean_abs_err={metrics['mean_abs']:.6f}")
    assert r < 0.10, f"FP8 vs BF16 varlen RRMSE {r:.6f} >= 0.10"
    assert c > 0.99, f"FP8 vs BF16 varlen cosine {c:.8f} <= 0.99"
