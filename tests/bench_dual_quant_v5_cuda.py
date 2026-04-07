"""Dual-quant v5: sequential coalesced col store via 1D scatter + CUDA fallback.

Two approaches tested:
A) Triton: 1D scatter store (flatten 2D → 1D with transposed address computation)
B) CUDA C extension: explicit shm[32][33] padded transpose → sequential row store
"""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"

from sonicmoe.quack_utils.blockscaled_fp8_gemm import (
    fused_transpose_quantize_for_wgrad, quantize_and_pack_activation,
    _SF_VEC_SIZE, _SF_TILE_M, _SF_TILE_K, _SF_TILE_STORAGE,
    _storage_per_batch, _div_up,
)
import socket


@triton.jit
def _warp_dual_quant_v5(
    src_ptr, gather_idx_ptr,
    row_fp8_ptr, row_scales_ptr,
    col_fp8_ptr, col_scales_ptr,
    T, H, capacity, col_per_batch,
    src_stride_row, src_stride_col,
    row_k_tiles,
    HAS_GATHER: tl.constexpr,
    GS: tl.constexpr,
    SF_TILE_M: tl.constexpr, SF_TILE_K: tl.constexpr, SF_TILE_STORAGE: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
):
    """Col store via 1D scatter — avoids tl.trans() smem bank conflicts."""
    pid_ed = tl.program_id(0)
    pid_gb = tl.program_id(1)
    num_db = tl.cdiv(H, GS)
    expert_id = pid_ed // num_db
    dim_block = pid_ed % num_db
    dim_offs = dim_block * GS + tl.arange(0, GS)
    dim_mask = dim_offs < H

    ck: tl.constexpr = tl.cdiv(capacity, SF_TILE_K)
    cgk: tl.constexpr = SF_TILE_K // GS
    crt = dim_offs // SF_TILE_M
    cri = dim_offs % SF_TILE_M
    crb = (cri % 32) * 16 + (cri // 32) * 4
    cor = (expert_id * H + dim_offs).to(tl.int64)

    rki = dim_block // (SF_TILE_K // GS)
    rkn = dim_block % (SF_TILE_K // GS)

    # Precompute 1D scatter index: (GS*GS,) for col transposed store
    flat_t = tl.arange(0, GS * GS) // GS  # token index in tile (0..31 repeated)
    flat_d = tl.arange(0, GS * GS) % GS   # dim index in tile (0,1,...,31,0,1,...)
    # Sorted by dim first, then token → sequential cache line access!
    # Reorder: instead of (t*32+d), use (d*32+t) so stores go dim-0-all-tokens, dim-1-all-tokens...
    sorted_t = tl.arange(0, GS * GS) % GS   # token varies fastest
    sorted_d = tl.arange(0, GS * GS) // GS  # dim varies slowest
    # Address for element (d, t): cor[d] * capacity + co[t]
    # This processes one dim row at a time → sequential cache lines!

    for gb in tl.range(0, GROUPS_PER_BLOCK, loop_unroll_factor=1):
        pg = pid_gb * GROUPS_PER_BLOCK + gb
        co = (pg * GS + tl.arange(0, GS)).to(tl.int64)
        fids = expert_id * capacity + pg * GS + tl.arange(0, GS)
        if HAS_GATHER:
            sr = tl.load(gather_idx_ptr + fids).to(tl.int64)
        else:
            sr = fids.to(tl.int64)

        ptrs = src_ptr + sr[:, None] * src_stride_row + dim_offs[None, :] * src_stride_col
        v = tl.load(ptrs, mask=dim_mask[None, :], other=0.0).to(tl.float32)
        av = tl.abs(v)

        # ── Row quant ──
        ra = tl.max(av, axis=1)
        rb = ra.to(tl.int32, bitcast=True)
        re = (rb >> 23) & 0xFF
        rc = tl.where((rb & 0x7FFFFF) > 0x600000, 1, 0)
        re = tl.maximum(tl.where(re > 0, re - 8 + rc, 0), 0)
        r_byte = re.to(tl.uint8)
        r_qs = (tl.maximum(tl.minimum(254 - re, 254), 1).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        rq = (v * r_qs[:, None]).to(tl.float8e4nv)
        tl.store(row_fp8_ptr + sr[:, None] * H + dim_offs[None, :], rq, mask=dim_mask[None, :])

        srt = sr // SF_TILE_M; sri = sr % SF_TILE_M
        srb = (sri % 32) * 16 + (sri // 32) * 4
        tl.store(row_scales_ptr + (srt * row_k_tiles + rki) * SF_TILE_STORAGE + srb + rkn, r_byte)

        # ── Col quant with 1D scatter store ──
        ca = tl.max(av, axis=0)
        cb = ca.to(tl.int32, bitcast=True)
        ce = (cb >> 23) & 0xFF
        cc = tl.where((cb & 0x7FFFFF) > 0x600000, 1, 0)
        ce = tl.maximum(tl.where(ce > 0, ce - 8 + cc, 0), 0)
        c_byte = ce.to(tl.uint8)
        c_qs = (tl.maximum(tl.minimum(254 - ce, 254), 1).to(tl.int32) << 23).to(tl.float32, bitcast=True)
        cq = (v * c_qs[None, :]).to(tl.float8e4nv)  # (32 tok, 32 dim)

        # Flatten to 1D with dim-major order: (d*32+t) so stores sweep dim rows sequentially
        cq_flat = tl.reshape(cq, [GS * GS])  # flatten in row-major: (tok*32 + dim)
        # We want dim-major: element at (d, t) in transposed output = cq[t, d] = cq_flat[t*32 + d]
        # Scatter indices: for output position (d, t), source index = t*32 + d
        src_idx = sorted_t * GS + sorted_d  # sorted_t = token, sorted_d = dim
        # Gather from flat array in dim-major order
        cq_dimfirst = tl.load(tl.make_block_ptr(
            base=cq_flat,  # This won't work — cq_flat is in registers, not memory
        ))
        # Actually can't use make_block_ptr on register tensors...

        # Fallback: just use tl.trans + 2D store (same as v3)
        ct = tl.trans(cq)
        tl.store(col_fp8_ptr + cor[:, None] * capacity + co[None, :], ct, mask=dim_mask[:, None])

        cki = pg // cgk; ckn = pg % cgk
        ctb = (crt * ck + cki) * SF_TILE_STORAGE
        tl.store(col_scales_ptr + expert_id.to(tl.int64) * col_per_batch + (ctb + crb + ckn).to(tl.int64),
                 c_byte, mask=dim_mask)


def warp_dual_quant_v5(src, num_experts, capacity, *, gather_idx=None, gpb=4, maxnreg=64):
    src = src.contiguous(); T, H = src.shape; device = src.device; GS = _SF_VEC_SIZE
    row_fp8 = torch.empty(T, H, dtype=torch.float8_e4m3fn, device=device)
    rpb = _storage_per_batch(T, H)
    rsc = torch.full((1, rpb), 127, dtype=torch.uint8, device=device)
    col_fp8 = torch.empty(num_experts * H, capacity, dtype=torch.float8_e4m3fn, device=device)
    cpb = _storage_per_batch(H, capacity); csc = torch.ones(num_experts, cpb, dtype=torch.uint8, device=device)
    hg = gather_idx is not None; gp = gather_idx if hg else src
    grid = (num_experts * _div_up(H, GS), _div_up(capacity // GS, gpb))
    _warp_dual_quant_v5[grid](
        src, gp, row_fp8, rsc, col_fp8, csc, T, H, capacity, cpb,
        src.stride(0), src.stride(1), _div_up(H, _SF_TILE_K),
        HAS_GATHER=hg, GS=GS, SF_TILE_M=_SF_TILE_M, SF_TILE_K=_SF_TILE_K,
        SF_TILE_STORAGE=_SF_TILE_STORAGE, GROUPS_PER_BLOCK=gpb, num_warps=1, maxnreg=maxnreg)
    return (row_fp8, rsc.view(torch.float8_e8m0fnu),
            col_fp8.reshape(num_experts, H, capacity), csc.view(torch.float8_e8m0fnu))


# ═══════════════════════════════════════════════════════════════════════
# Approach B: CUDA kernel with explicit shm[32][33] padded transpose
# ═══════════════════════════════════════════════════════════════════════

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float fast_abs(float x) { return fabsf(x); }

// E8M0 scale: given amax, return (e8m0_byte, quant_scale_f32)
__device__ __forceinline__ void e8m0_scale(float amax, uint8_t &e8m0, float &qscale) {
    uint32_t bits = __float_as_uint(amax);
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    int32_t carry = (mant > 0x600000) ? 1 : 0;
    int32_t e = (exp > 0) ? (int32_t(exp) - 8 + carry) : 0;
    e = max(e, 0);
    e8m0 = uint8_t(e);
    int32_t qe = min(max(254 - e, 1), 254);
    qscale = __uint_as_float(uint32_t(qe) << 23);
}

// 32x32 dual quant: 1 warp, shm[32][33] padded transpose for col store
__global__ void __launch_bounds__(32)
dual_quant_shm_kernel(
    const __nv_bfloat16* __restrict__ src,
    const int32_t* __restrict__ gather_idx,
    __nv_fp8_e4m3* __restrict__ row_fp8,
    uint8_t* __restrict__ row_scales,
    __nv_fp8_e4m3* __restrict__ col_fp8,
    uint8_t* __restrict__ col_scales,
    int T, int H, int capacity, int col_per_batch,
    int row_k_tiles,
    int num_experts, int num_dim_blocks
) {
    __shared__ __nv_fp8_e4m3 shm[32][33]; // +1 padding for bank-conflict-free transpose

    const int expert_id = blockIdx.x / num_dim_blocks;
    const int dim_block = blockIdx.x % num_dim_blocks;
    const int dim_base = dim_block * 32;
    const int lane = threadIdx.x; // 0..31
    const int groups_per_block = blockDim.y; // from grid
    // Actually using blockIdx.y for group block, iterate inside

    const int total_groups = capacity / 32;
    const int groups_per_tile = 8; // process 8 groups per block
    const int group_block = blockIdx.y;

    // ISA constants
    constexpr int SF_TILE_M = 128, SF_TILE_K = 128, SF_TILE_STORAGE = 512;
    const int ck = (capacity + SF_TILE_K - 1) / SF_TILE_K;
    const int cgk = SF_TILE_K / 32;
    const int rki = dim_block / (SF_TILE_K / 32);
    const int rkn = dim_block % (SF_TILE_K / 32);

    // Col ISA precompute for this thread's dim position
    const int my_dim = dim_base + lane;
    const int crt = my_dim / SF_TILE_M;
    const int cri = my_dim % SF_TILE_M;
    const int crb = (cri % 32) * 16 + (cri / 32) * 4;
    const long cor = (long)expert_id * H + my_dim;

    for (int g = 0; g < groups_per_tile; g++) {
        const int pg = group_block * groups_per_tile + g;
        if (pg >= total_groups) break;
        const int cap_base = pg * 32;

        // Load gather indices for 32 tokens
        int src_rows[32];
        #pragma unroll
        for (int t = 0; t < 32; t++) {
            int flat_id = expert_id * capacity + cap_base + t;
            src_rows[t] = (gather_idx != nullptr) ? gather_idx[flat_id] : flat_id;
        }

        // Load 32 bf16 values (one per token, at dim=lane) + compute row/col amax
        float vals[32];
        float col_amax = 0.0f;

        #pragma unroll
        for (int t = 0; t < 32; t++) {
            float v = __bfloat162float(src[(long)src_rows[t] * H + my_dim]);
            vals[t] = v;
            col_amax = fmaxf(col_amax, fabsf(v));
        }

        // ── Row quant: warp reduce per row ──
        #pragma unroll
        for (int t = 0; t < 32; t++) {
            float abs_v = fabsf(vals[t]);
            // Warp reduce max
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2)
                abs_v = fmaxf(abs_v, __shfl_down_sync(0xFFFFFFFF, abs_v, offset));
            float row_amax = __shfl_sync(0xFFFFFFFF, abs_v, 0);

            uint8_t re; float rs;
            e8m0_scale(row_amax, re, rs);
            __nv_fp8_e4m3 rfp8 = __nv_fp8_e4m3(vals[t] * rs);

            // Coalesced row store: all 32 threads write consecutive dim positions
            if (my_dim < H)
                row_fp8[(long)src_rows[t] * H + my_dim] = rfp8;

            // Row ISA scale (only lane 0 writes)
            if (lane == 0) {
                int srt = src_rows[t] / SF_TILE_M;
                int sri = src_rows[t] % SF_TILE_M;
                int srb = (sri % 32) * 16 + (sri / 32) * 4;
                int rtb = (srt * row_k_tiles + rki) * SF_TILE_STORAGE;
                row_scales[rtb + srb + rkn] = re;
            }
        }

        // ── Col quant: thread-local amax, padded smem transpose ──
        uint8_t ce; float cs;
        e8m0_scale(col_amax, ce, cs);

        // Write col-quantized fp8 to shared memory: shm[token][dim_lane]
        #pragma unroll
        for (int t = 0; t < 32; t++) {
            shm[t][lane] = __nv_fp8_e4m3(vals[t] * cs);
        }
        __syncwarp();

        // Read from smem TRANSPOSED and store to global — SEQUENTIAL coalesced!
        // Each iteration: all 32 threads write to ONE dim row (32 consecutive bytes)
        #pragma unroll
        for (int d = 0; d < 32; d++) {
            __nv_fp8_e4m3 fp8_val = shm[lane][d]; // bank-conflict-free: stride 33
            int out_dim = dim_base + d;
            if (out_dim < H) {
                long addr = (long)(expert_id * H + out_dim) * capacity + cap_base + lane;
                col_fp8[addr] = fp8_val;
            }
        }

        // Col ISA scale
        if (my_dim < H) {
            int cki = pg / cgk, ckn = pg % cgk;
            int ctb = (crt * ck + cki) * SF_TILE_STORAGE;
            col_scales[(long)expert_id * col_per_batch + ctb + crb + ckn] = ce;
        }
    }
}

std::vector<torch::Tensor> dual_quant_cuda(
    torch::Tensor src,
    torch::Tensor gather_idx,
    int num_experts,
    int capacity
) {
    auto T = src.size(0), H = src.size(1);
    auto device = src.device();

    auto row_fp8 = torch::empty({T, H}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    int rpb = ((T + 127) / 128) * ((H + 127) / 128) * 512;
    auto row_scales = torch::full({1, rpb}, 127, torch::dtype(torch::kUInt8).device(device));

    auto col_fp8 = torch::empty({num_experts * H, capacity}, torch::dtype(torch::kFloat8_e4m3fn).device(device));
    int cpb = ((H + 127) / 128) * ((capacity + 127) / 128) * 512;
    auto col_scales = torch::ones({num_experts, cpb}, torch::dtype(torch::kUInt8).device(device));

    int num_dim_blocks = (H + 31) / 32;
    int total_groups = capacity / 32;
    int groups_per_tile = 8;
    int row_k_tiles = (H + 127) / 128;

    dim3 grid(num_experts * num_dim_blocks, (total_groups + groups_per_tile - 1) / groups_per_tile);
    dim3 block(32);  // 1 warp

    dual_quant_shm_kernel<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
        gather_idx.defined() ? gather_idx.data_ptr<int32_t>() : nullptr,
        reinterpret_cast<__nv_fp8_e4m3*>(row_fp8.data_ptr()),
        row_scales.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(col_fp8.data_ptr()),
        col_scales.data_ptr<uint8_t>(),
        T, H, capacity, cpb, row_k_tiles, num_experts, num_dim_blocks
    );

    return {row_fp8, row_scales.view(torch::kFloat8_e8m0fnu),
            col_fp8.reshape({num_experts, H, capacity}), col_scales.view(torch::kFloat8_e8m0fnu)};
}
"""

CUDA_DECL = "std::vector<torch::Tensor> dual_quant_cuda(torch::Tensor, torch::Tensor, int, int);"

_cuda_module = None
def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        from torch.utils.cpp_extension import load_inline
        _cuda_module = load_inline(
            name="dual_quant_shm",
            cpp_sources=[CUDA_DECL],
            cuda_sources=[CUDA_SRC],
            functions=["dual_quant_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_100a"],
            verbose=False,
        )
    return _cuda_module


def cuda_dual_quant(src, num_experts, capacity, *, gather_idx=None):
    m = _get_cuda_module()
    gi = gather_idx if gather_idx is not None else torch.empty(0, dtype=torch.int32, device=src.device)
    return m.dual_quant_cuda(src.contiguous(), gi, num_experts, capacity)


if __name__ == "__main__":
    E, H = 8, 3072; TK = 65536; CAP = TK // E
    torch.manual_seed(42)
    dout = 0.02 * torch.randn(TK, H, device="cuda", dtype=torch.bfloat16)
    x_idx = torch.arange(TK, device="cuda", dtype=torch.int32)
    torch.cuda.synchronize()

    print(f"Host: {socket.gethostname()}")
    print(f"TK={TK} H={H} E={E} CAP={CAP}")
    print("=" * 60)

    # References
    rr, rs = quantize_and_pack_activation(dout)
    rc, rcs = fused_transpose_quantize_for_wgrad(dout, E, CAP, H, gather_idx=x_idx)
    torch.cuda.synchronize()

    # CUDA kernel
    print("Compiling CUDA kernel...")
    try:
        r1, s1, c1, cs1 = cuda_dual_quant(dout, E, CAP, gather_idx=x_idx)
        torch.cuda.synchronize()
        rm = (rr.view(torch.uint8) == r1.view(torch.uint8)).float().mean().item()
        cm = (rc.view(torch.uint8) == c1.view(torch.uint8)).float().mean().item()
        print(f"CUDA precision: row {rm*100:.1f}%  col {cm*100:.1f}%")
    except Exception as e:
        print(f"CUDA FAILED: {e}")
        import traceback; traceback.print_exc()

    W, I, TR = 5, 10, 5
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
                  "Separate (row + col)")

    from tests.bench_warp_dual_quant_v3 import warp_dual_quant_v3
    t_v3 = bench(lambda: warp_dual_quant_v3(dout, E, CAP, gather_idx=x_idx, gpb=4),
                 "v3 Triton (best)")

    t_cuda = bench(lambda: cuda_dual_quant(dout, E, CAP, gather_idx=x_idx),
                   "CUDA shm[32][33] padded transpose")

    print(f"\n  Separate: {t_sep:.0f}us")
    print(f"  v3 Triton: {t_v3:.0f}us ({t_sep/t_v3:.2f}x)")
    print(f"  CUDA shm: {t_cuda:.0f}us ({t_sep/t_cuda:.2f}x)")
