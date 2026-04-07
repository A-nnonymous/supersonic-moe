"""CUDA dual-quant v6: 8 warps, hardware redux, shm[32][33] padded transpose.

Key design:
- 8 warps per block (256 threads), __launch_bounds__(256)
- Each warp processes one 32×32 group independently
- All 8 warps write to SAME 32 dim rows → L1 cache lines stay warm
- Row amax: __reduce_max_sync via int32 bitcast (1 instruction vs 5 shfl)
- Col store: shm[warp][32][33] padded → sequential coalesced global store
- Grid: (E * ceil(H/32), ceil(total_groups/8))
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    fused_transpose_quantize_for_wgrad, quantize_and_pack_activation,
    _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE,
    _storage_per_batch, _div_up,
)
import socket

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

__device__ __forceinline__ void e8m0_scale(float amax, uint8_t &e8m0, float &qscale) {
    uint32_t bits = __float_as_uint(amax);
    uint32_t exp = (bits >> 23) & 0xFF;
    int32_t e = (exp > 0) ? (int32_t(exp) - 8 + (((bits & 0x7FFFFF) > 0x600000) ? 1 : 0)) : 0;
    e = max(e, 0);
    e8m0 = uint8_t(e);
    qscale = __uint_as_float(uint32_t(min(max(254 - e, 1), 254)) << 23);
}

// Hardware warp max reduction for positive floats via int32 bitcast
__device__ __forceinline__ float warp_reduce_max_abs(float val) {
    // For |val| >= 0, float bit pattern preserves order as uint32
    uint32_t ival = __float_as_uint(val);
    // 5 shuffle steps (works on all SM)
    ival = max(ival, __shfl_xor_sync(0xFFFFFFFF, ival, 16));
    ival = max(ival, __shfl_xor_sync(0xFFFFFFFF, ival, 8));
    ival = max(ival, __shfl_xor_sync(0xFFFFFFFF, ival, 4));
    ival = max(ival, __shfl_xor_sync(0xFFFFFFFF, ival, 2));
    ival = max(ival, __shfl_xor_sync(0xFFFFFFFF, ival, 1));
    return __uint_as_float(ival);
}

// 8-warp dual quant: each warp handles one 32x32 group
// All warps in a block write to the SAME 32 dim rows → L1 stays warm
__global__ void __launch_bounds__(256, 2)
dual_quant_8warp_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ gather_idx,
    __nv_fp8_e4m3* __restrict__ row_fp8,
    uint8_t* __restrict__ row_scales,
    __nv_fp8_e4m3* __restrict__ col_fp8,
    uint8_t* __restrict__ col_scales,
    int T, int H, int capacity, int col_per_batch,
    int row_k_tiles, int num_dim_blocks, int total_groups,
    bool has_gather
) {
    // shm: 8 warps × 32 rows × 33 cols (padded for bank-conflict-free transpose)
    __shared__ __nv_fp8_e4m3 shm[8][32][33];

    const int expert_id = blockIdx.x / num_dim_blocks;
    const int dim_block = blockIdx.x % num_dim_blocks;
    const int dim_base = dim_block * 32;
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int my_dim = dim_base + lane;

    // Each warp processes one group
    const int pg = blockIdx.y * 8 + warp_id;
    if (pg >= total_groups) return;
    const int cap_base = pg * 32;

    // ISA constants
    constexpr int SF_M = 128, SF_K = 128, SF_S = 512;
    const int ck = (capacity + SF_K - 1) / SF_K;
    const int cgk = SF_K / 32;
    const int rki = dim_block / (SF_K / 32);
    const int rkn = dim_block % (SF_K / 32);

    // Load 32 source rows for this warp's group
    float vals[32];
    float col_amax = 0.0f;
    int src_rows[32];

    #pragma unroll
    for (int t = 0; t < 32; t++) {
        int flat_id = expert_id * capacity + cap_base + t;
        src_rows[t] = has_gather ? gather_idx[flat_id] : flat_id;
        float v = (my_dim < H) ? __bfloat162float(src[(long)src_rows[t] * H + my_dim]) : 0.0f;
        vals[t] = v;
        col_amax = fmaxf(col_amax, fabsf(v));
    }

    // ── Row quant: warp reduce per row ──
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        float row_amax = warp_reduce_max_abs(fabsf(vals[t]));
        uint8_t re; float rs;
        e8m0_scale(row_amax, re, rs);
        __nv_fp8_e4m3 rfp8 = __nv_fp8_e4m3(vals[t] * rs);

        // Coalesced row store (32 consecutive dim positions per warp)
        if (my_dim < H)
            row_fp8[(long)src_rows[t] * H + my_dim] = rfp8;

        // Row ISA scale (lane 0 per warp)
        if (lane == 0) {
            int sr = src_rows[t];
            int srt = sr / SF_M, sri = sr % SF_M;
            int srb = (sri % 32) * 16 + (sri / 32) * 4;
            row_scales[(srt * row_k_tiles + rki) * SF_S + srb + rkn] = re;
        }
    }

    // ── Col quant: thread-local amax → smem padded transpose → coalesced store ──
    uint8_t ce; float cs;
    e8m0_scale(col_amax, ce, cs);

    // Write col-quantized fp8 to padded shared memory
    #pragma unroll
    for (int t = 0; t < 32; t++) {
        shm[warp_id][t][lane] = __nv_fp8_e4m3(vals[t] * cs);
    }
    __syncwarp();

    // Read transposed from smem → sequential coalesced global store
    // Each iteration: warp writes 32 consecutive bytes at one dim row
    #pragma unroll
    for (int d = 0; d < 32; d++) {
        __nv_fp8_e4m3 fp8_val = shm[warp_id][lane][d]; // bank-conflict-free (stride 33)
        int out_dim = dim_base + d;
        if (out_dim < H) {
            col_fp8[(long)(expert_id * H + out_dim) * capacity + cap_base + lane] = fp8_val;
        }
    }

    // Col ISA scale (one per dim position per group)
    if (my_dim < H) {
        int crt = my_dim / SF_M, cri = my_dim % SF_M;
        int crb = (cri % 32) * 16 + (cri / 32) * 4;
        int cki = pg / cgk, ckn = pg % cgk;
        col_scales[(long)expert_id * col_per_batch + (crt * ck + cki) * SF_S + crb + ckn] = ce;
    }
}

std::vector<torch::Tensor> dual_quant_8warp(
    torch::Tensor src, torch::Tensor gather_idx, int num_experts, int capacity
) {
    int T = src.size(0), H = src.size(1);
    auto dev = src.device();
    auto row_fp8 = torch::empty({T, H}, torch::dtype(torch::kFloat8_e4m3fn).device(dev));
    int rpb = ((T+127)/128) * ((H+127)/128) * 512;
    auto row_sc = torch::full({1, rpb}, 127, torch::dtype(torch::kUInt8).device(dev));
    auto col_fp8 = torch::empty({num_experts*H, capacity}, torch::dtype(torch::kFloat8_e4m3fn).device(dev));
    int cpb = ((H+127)/128) * ((capacity+127)/128) * 512;
    auto col_sc = torch::ones({num_experts, cpb}, torch::dtype(torch::kUInt8).device(dev));

    int ndb = (H + 31) / 32;
    int tg = capacity / 32;
    int rkt = (H + 127) / 128;
    bool hg = gather_idx.defined() && gather_idx.numel() > 0;

    dim3 grid(num_experts * ndb, (tg + 7) / 8);
    dim3 block(256);  // 8 warps

    dual_quant_8warp_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
        hg ? gather_idx.data_ptr<int32_t>() : nullptr,
        reinterpret_cast<__nv_fp8_e4m3*>(row_fp8.data_ptr()),
        row_sc.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(col_fp8.data_ptr()),
        col_sc.data_ptr<uint8_t>(),
        T, H, capacity, cpb, rkt, ndb, tg, hg
    );
    return {row_fp8, row_sc.view(torch::kFloat8_e8m0fnu),
            col_fp8.reshape({num_experts, H, capacity}), col_sc.view(torch::kFloat8_e8m0fnu)};
}
"""

_mod = None
def _get():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(name="dq8w", cpp_sources=[
            "std::vector<torch::Tensor> dual_quant_8warp(torch::Tensor,torch::Tensor,int,int);"
        ], cuda_sources=[CUDA_SRC], functions=["dual_quant_8warp"],
           extra_cuda_cflags=["-O3","--use_fast_math","-arch=sm_100a"], verbose=False)
    return _mod


def cuda_dual_quant_8warp(src, num_experts, capacity, *, gather_idx=None):
    gi = gather_idx if gather_idx is not None else torch.empty(0, dtype=torch.int32, device=src.device)
    return _get().dual_quant_8warp(src.contiguous(), gi, num_experts, capacity)


if __name__ == "__main__":
    E, H = 8, 3072; TK = 65536; CAP = TK // E
    torch.manual_seed(42)
    dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
    x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()

    print(f"Host: {socket.gethostname()}")
    print(f"TK={TK} H={H} E={E} CAP={CAP}")
    print("=" * 60)

    rr, _ = quantize_and_pack_activation(dout)
    rc, _ = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
    torch.cuda.synchronize()

    print("Compiling 8-warp CUDA kernel...")
    r1, _, c1, _ = cuda_dual_quant_8warp(dout, E, CAP, gather_idx=x_idx)
    torch.cuda.synchronize()
    rm = (rr.view(torch.uint8) == r1.view(torch.uint8)).float().mean().item()
    cm = (rc.view(torch.uint8) == c1.view(torch.uint8)).float().mean().item()
    print(f"Precision: row {rm*100:.1f}%  col {cm*100:.1f}%")

    W, I, TR = 10, 20, 5
    def bench(fn, name):
        for _ in range(W): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(TR):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(I): fn()
            e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1000 / I)
        print(f"  {name:<55} min={min(ts):>7.0f}us")
        return min(ts)

    print("\n--- Benchmark ---")
    t_sep = bench(lambda: (quantize_and_pack_activation(dout),
                           fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)),
                  "Separate (Triton row + col)")
    t_cuda = bench(lambda: cuda_dual_quant_8warp(dout, E, CAP, gather_idx=x_idx),
                   "CUDA 8-warp shm[32][33]")
    print(f"\n  {t_sep:.0f}us → {t_cuda:.0f}us ({t_sep/t_cuda:.2f}x)")
