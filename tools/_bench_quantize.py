#!/usr/bin/env python3
"""Microbenchmark for blockscaled quantize-and-pack kernels.

Tests different BLOCK_ROWS and kernel variants.
"""
import torch
import time
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("USE_QUACK_GEMM", "1")

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    quantize_and_pack_activation,
    pad_quantize_and_pack_activation,
    gather_quantize_and_pack_activation,
    _div_up,
    _SF_TILE_M,
)

def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def main():
    device = "cuda"
    M, K = 32768, 4096

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    print(f"Shape: M={M}, K={K}")
    print(f"Theoretical bandwidth limit: {(M*K*2 + M*K*1 + M*(K//32))/(8e9):.3f}ms (B200 ~8TB/s)")
    print()

    # 1. quantize_and_pack_activation
    t = bench(lambda: quantize_and_pack_activation(x))
    print(f"quantize_and_pack_activation:        {t:.3f}ms")

    # 2. gather_quantize_and_pack_activation
    gather_idx = torch.randperm(M, device=device, dtype=torch.int64)
    src = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    t = bench(lambda: gather_quantize_and_pack_activation(src, gather_idx))
    print(f"gather_quantize_and_pack_activation: {t:.3f}ms")

    # 3. pad_quantize_and_pack_activation (simulate 25% padding)
    padded_total = _div_up(M, _SF_TILE_M) * _SF_TILE_M + _SF_TILE_M * (M // (_SF_TILE_M * 4))
    padded_total = min(padded_total, int(M * 1.25))
    padded_total = _div_up(padded_total, _SF_TILE_M) * _SF_TILE_M
    dst_idx = torch.arange(M, device=device, dtype=torch.int64)
    dst_idx = dst_idx + (dst_idx // (_SF_TILE_M * 4)) * _SF_TILE_M
    dst_idx = dst_idx[:M].clamp(max=padded_total - 1)
    t = bench(lambda: pad_quantize_and_pack_activation(x, padded_total, dst_idx))
    print(f"pad_quantize_and_pack_activation:    {t:.3f}ms  (padded_total={padded_total})")

    # 4. SwiGLU fused quant kernels
    from sonicmoe.quack_utils.swiglu_triton import (
        swiglu_forward_quant_pack_triton,
        swiglu_backward_quant_pack_triton,
    )
    TK, I = M, K // 4  # I=1024 for K=4096, z has 2I=2048 cols
    z = torch.randn(TK, 2 * I, device=device, dtype=torch.bfloat16)
    dy1 = torch.randn(TK, I, device=device, dtype=torch.bfloat16)
    s = torch.rand(TK, device=device, dtype=torch.float32)

    t = bench(lambda: swiglu_forward_quant_pack_triton(z))
    print(f"swiglu_fwd_quant_pack:               {t:.3f}ms")

    t = bench(lambda: swiglu_backward_quant_pack_triton(dy1, z, s))
    print(f"swiglu_bwd_quant_pack:               {t:.3f}ms")

if __name__ == "__main__":
    main()
