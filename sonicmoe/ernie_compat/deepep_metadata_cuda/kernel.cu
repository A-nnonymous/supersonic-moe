// DeepEP → SonicMoE metadata fill kernel (CUDA) — V2 optimized.
//
// V1 NCU findings:
//   - Kernel GPU compute: 1,372 SM cycles ≈ 1.25 μs (already fast)
//   - End-to-end: 320-450 μs due to Python overhead
//   - Barrier stall 31.6%: __syncthreads between serial prefix-sum and fill
//   - 1 block on 148-SM GPU: 0.00 waves, 9.4% achieved occupancy
//
// V2 optimization:
//   - Prefix-sum moved to host (tokens_per_expert is already a host list)
//   - Host passes pre-computed per-expert metadata as device tensors
//   - 1 block per expert (E blocks total) → no barriers, better SM utilization
//   - Fused int4+float4 stores: x_gather_idx and router_scores written together
//   - No item() DtoH sync → eliminates the dominant Python overhead
//
// Design from Paddle's moe_permute_kernel.cu:
//   - Per-block segment ownership (each block owns one expert's output range)
//   - Vectorized int4/float4 global stores for bulk fill

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK_DIM 256

// ============================================================================
//  Per-expert fill kernel: 1 block per expert, no synchronization needed
// ============================================================================
//
// Each block fills x_gather_idx[seg_start..seg_start+seg_len) and
// router_scores[seg_start..seg_start+seg_len) for one expert.
//
// Inputs (per-expert, pre-computed on host):
//   seg_starts[e]  = padded cumulative offset for expert e
//   seg_lens[e]    = padded segment length for expert e
//   real_counts[e] = actual token count for expert e
//   real_bases[e]  = cumulative real token offset for expert e
//   pad_bases[e]   = T + cumulative padding offset for expert e
//
__global__ __launch_bounds__(BLOCK_DIM)
void deepep_fill_kernel(
    int*       __restrict__ x_gather_idx,    // [TK_padded] output
    float*     __restrict__ router_scores,   // [TK_padded] output
    const int* __restrict__ seg_starts,      // [E] pre-computed
    const int* __restrict__ seg_lens,        // [E] pre-computed
    const int* __restrict__ real_counts,     // [E] pre-computed
    const int* __restrict__ real_bases,      // [E] pre-computed
    const int* __restrict__ pad_bases)       // [E] pre-computed
{
    const int e = blockIdx.x;
    const int seg_start = seg_starts[e];
    const int seg_len   = seg_lens[e];
    if (seg_len == 0) return;

    const int count     = real_counts[e];
    const int real_base = real_bases[e];
    // pad_bases[e] is unused — route-level padding uses gather index 0.

    // All segments are 128-aligned → seg_len is always divisible by 4.
    // Use fused int4 + float4 writes: each thread writes 4 gather indices
    // and 4 router scores per iteration, coalesced across the warp.
    const int seg_len_vec4 = seg_len >> 2;

    int*   g_idx = x_gather_idx + seg_start;
    float* g_scr = router_scores + seg_start;

    for (int vi = threadIdx.x; vi < seg_len_vec4; vi += BLOCK_DIM) {
        const int base_j = vi << 2;

        int4 idx_vec;
        float4 scr_vec;
        int*   iv = reinterpret_cast<int*>(&idx_vec);
        float* sv = reinterpret_cast<float*>(&scr_vec);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int j = base_j + k;
            const bool is_real = (j < count);
            // Route-level padding: padding rows gather from row 0 (score=0
            // nullifies contribution).  Matches frontier _pad_routing_metadata.
            iv[k] = is_real ? (real_base + j) : 0;
            sv[k] = is_real ? 1.0f : 0.0f;
        }

        reinterpret_cast<int4*>(g_idx)[vi] = idx_vec;
        reinterpret_cast<float4*>(g_scr)[vi] = scr_vec;
    }

    // Scalar tail (only when seg_len % 4 != 0 — shouldn't happen with
    // 128-alignment but handle defensively)
    for (int j = (seg_len_vec4 << 2) + threadIdx.x; j < seg_len; j += BLOCK_DIM) {
        const bool is_real = (j < count);
        g_idx[j] = is_real ? (real_base + j) : 0;
        g_scr[j] = is_real ? 1.0f : 0.0f;
    }
}


// ============================================================================
//  C++ interface — V2: host passes pre-computed metadata, no item() needed
// ============================================================================
void deepep_metadata_cuda(
    torch::Tensor& expert_freq_offset,  // [E+1] int32 (host-computed, pre-on-device)
    torch::Tensor& x_gather_idx,        // [TK_padded] int32
    torch::Tensor& router_scores,       // [TK_padded] float32
    torch::Tensor& seg_starts,          // [E] int32 (device)
    torch::Tensor& seg_lens,            // [E] int32 (device)
    torch::Tensor& real_counts,         // [E] int32 (device)
    torch::Tensor& real_bases,          // [E] int32 (device)
    torch::Tensor& pad_bases,           // [E] int32 (device)
    int64_t E,
    int64_t stream_ptr)
{
    if (E == 0) return;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    deepep_fill_kernel<<<static_cast<int>(E), BLOCK_DIM, 0, stream>>>(
        x_gather_idx.data_ptr<int>(),
        router_scores.data_ptr<float>(),
        seg_starts.data_ptr<int>(),
        seg_lens.data_ptr<int>(),
        real_counts.data_ptr<int>(),
        real_bases.data_ptr<int>(),
        pad_bases.data_ptr<int>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deepep_metadata_cuda",
          &deepep_metadata_cuda,
          "DeepEP metadata fill (CUDA) — V2 multi-block",
          py::arg("expert_freq_offset"),
          py::arg("x_gather_idx"),
          py::arg("router_scores"),
          py::arg("seg_starts"),
          py::arg("seg_lens"),
          py::arg("real_counts"),
          py::arg("real_bases"),
          py::arg("pad_bases"),
          py::arg("E"),
          py::arg("stream"));
}
