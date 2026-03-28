#!/usr/bin/env python3
"""Minimal test to understand tl.dot_scaled behavior on SM100."""
import torch
import triton
import triton.language as tl


@triton.jit
def test_dot_scaled_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scale_ptr, B_scale_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    GROUPS: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :])

    g_offs = tl.arange(0, GROUPS)
    a_s = tl.load(A_scale_ptr + offs_m[:, None] * GROUPS + g_offs[None, :])
    b_s = tl.load(B_scale_ptr + offs_n[:, None] * GROUPS + g_offs[None, :])

    c = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3")
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], c.to(tl.bfloat16))


@triton.jit
def test_regular_dot_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.dot(a, b).to(tl.bfloat16)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], c)


@triton.jit
def test_dot_scaled_bt_kernel(
    A_ptr, B_ptr, C_ptr,
    A_scale_ptr, B_scale_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    GROUPS: tl.constexpr,
):
    """B stored as (N, K) in memory, load transposed as (K, N)."""
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(A_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B_ptr + offs_k[:, None] + offs_n[None, :] * K)

    g_offs = tl.arange(0, GROUPS)
    a_s = tl.load(A_scale_ptr + offs_m[:, None] * GROUPS + g_offs[None, :])
    b_s = tl.load(B_scale_ptr + offs_n[:, None] * GROUPS + g_offs[None, :])

    c = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3")
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], c.to(tl.bfloat16))


def relrmse(a, b):
    diff = a.float() - b.float()
    return ((diff ** 2).mean().sqrt() / ((b.float() ** 2).mean().sqrt() + 1e-6)).item()


def main():
    M, N, K = 32, 32, 64
    device = "cuda"

    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    b_bf16 = torch.randn(K, N, dtype=torch.bfloat16, device=device) * 0.5
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)

    # scale=1 in E8M0: byte 127 = 2^(127-127) = 1.0
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device=device)
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device)

    # Reference: FP32 matmul of FP8 values
    c_ref = (a_fp8.float() @ b_fp8.float()).to(torch.bfloat16)

    # Test 1: regular tl.dot
    c_regular = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    test_regular_dot_kernel[(1,)](a_fp8, b_fp8, c_regular, M=M, N=N, K=K)
    print(f"regular_dot vs ref:          relrmse={relrmse(c_regular, c_ref):.6f}")

    # Test 2: dot_scaled with scale=1, B as (K, N)
    c_scaled = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    test_dot_scaled_kernel[(1,)](a_fp8, b_fp8, c_scaled, a_scale, b_scale, M=M, N=N, K=K, GROUPS=K//32)
    print(f"dot_scaled(B=KN, scale=1):   relrmse={relrmse(c_scaled, c_ref):.6f}")

    # Test 3: dot_scaled with B stored as (N, K)
    b_fp8_NK = b_fp8.T.contiguous()
    c_bt = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    test_dot_scaled_bt_kernel[(1,)](a_fp8, b_fp8_NK, c_bt, a_scale, b_scale, M=M, N=N, K=K, GROUPS=K//32)
    print(f"dot_scaled(B=NK trans, s=1):  relrmse={relrmse(c_bt, c_ref):.6f}")

    print(f"\nRef [0,:5]:     {c_ref[0,:5].tolist()}")
    print(f"Regular [0,:5]: {c_regular[0,:5].tolist()}")
    print(f"Scaled  [0,:5]: {c_scaled[0,:5].tolist()}")
    print(f"BT      [0,:5]: {c_bt[0,:5].tolist()}")

    # Test 4: dot_scaled with non-trivial scales
    print("\n--- Non-trivial scale test ---")
    # scale byte 120 = 2^(120-127) = 2^(-7) = 1/128
    a_scale2 = torch.full((M, K // 32), 120, dtype=torch.uint8, device=device)
    b_scale2 = torch.full((N, K // 32), 127, dtype=torch.uint8, device=device)  # 1.0
    c_s2 = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    test_dot_scaled_kernel[(1,)](a_fp8, b_fp8, c_s2, a_scale2, b_scale2, M=M, N=N, K=K, GROUPS=K//32)

    # Expected: c_ref * 1/128 (since a_scale applies per-group)
    expected_factor = 2.0 ** (-7)
    c_expected = (c_ref.float() * expected_factor).to(torch.bfloat16)
    print(f"Expected factor: {expected_factor}")
    print(f"dot_scaled(a_s=2^-7) vs ref*2^-7: relrmse={relrmse(c_s2, c_expected):.6f}")
    print(f"Expected [0,:3]:   {c_expected[0,:3].tolist()}")
    print(f"Got      [0,:3]:   {c_s2[0,:3].tolist()}")


if __name__ == "__main__":
    main()
