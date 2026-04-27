// DeepEP topk -> SonicMoE metadata: high-performance 2-kernel design.
//
// Eliminates ALL spin-wait. Uses histogram-based prefix sum approach:
//   Kernel 1: histogram (per-block expert counts) + naept counts + prefix sums
//   Kernel 2: scatter + fixup (fully parallel, zero synchronization)
//
// Key optimizations vs v1 (single-kernel with inter-block chain):
//   - No spin-wait: 94.2% "No Eligible" stall eliminated entirely
//   - Single atomic flag barrier (1 global read) instead of 512-deep chain
//   - Fully coalesced reads of dispatched_indices/probs (row-major, 32 rows/block)
//   - Warp ballot + __popc for stable intra-block ordering (identical to moe_permute)
//   - Template-unrolled topk loop
//   - Vectorized int4 pad-fill in fixup
//   - Binary search for expert lookup in fixup (O(log E))
//
// Token ordering: STABLE ascending within each expert (identical to moe_permute).

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

static constexpr int ROWS_PER_BLOCK = 32;  // = warp size for ballot
static constexpr int BLOCK_DIM = 256;
static constexpr int MAX_TOPK = 16;

// ============================================================================
//  Kernel 1a: Histogram only (Phase A)
//
//  Each block independently counts per-expert tokens in its 32 rows using warp
//  ballot → writes block_hist[blockIdx.x * E + e]; also writes
//  naept[row+1] = per-token valid count.
//
//  No inter-block synchronization — safe for arbitrarily large grids.
//  Phase B (prefix sums) runs as a separate kernel launch on the same stream,
//  giving an implicit grid barrier.  This avoids the ad-hoc grid spin-wait
//  that deadlocks once `scatter_blocks` exceeds device occupancy (B30Z hangs
//  at scatter_blocks ≈ 1358, observed via cuda-gdb on the legacy combined
//  kernel at kernel.cu:121).
// ============================================================================
template <int TOPK>
__global__ __launch_bounds__(BLOCK_DIM)
void histogram_kernel(
    const int*   __restrict__ dispatched_indices,   // [N_recv, topk]
    int*         __restrict__ block_hist,            // [scatter_blocks * E] output
    int*         __restrict__ naept,                 // [N_recv+1] output (entries [1..N_recv])
    int          N_recv,
    int          num_experts,
    int          topk_param)
{
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int warp_num = BLOCK_DIM >> 5;

    const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int global_row = block_row_base + lane_id;
    const bool row_valid = global_row < N_recv;

    extern __shared__ char smem[];
    uint32_t* expert_bitmask = reinterpret_cast<uint32_t*>(smem);

    // Initialize bitmask
    for (int i = threadIdx.x; i < num_experts; i += BLOCK_DIM) {
        expert_bitmask[i] = 0u;
    }
    __syncthreads();

    // Load topk entries + build per-expert bitmask
    int reg_valid_count = 0;

    #pragma unroll
    for (int col = 0; col < TOPK; col++) {
        if (col >= topk_param) break;
        int expert = -1;
        if (row_valid) {
            expert = dispatched_indices[global_row * topk_param + col];
        }
        if (expert >= 0 && expert < num_experts) {
            // Use warp-distributed atomicOr to reduce contention
            if (col % warp_num == warp_id) {
                atomicOr(&expert_bitmask[expert], 1u << lane_id);
            }
            reg_valid_count++;
        }
    }
    __syncthreads();

    // Write naept per-token count
    if (row_valid) {
        naept[global_row + 1] = reg_valid_count;
    }

    // Write per-expert counts for this block to block_hist
    for (int e = warp_id; e < num_experts; e += warp_num) {
        int count = __popc(expert_bitmask[e]);
        if (lane_id == 0) {
            block_hist[blockIdx.x * num_experts + e] = count;
        }
    }
}

// ============================================================================
//  Kernel 1b: Prefix sums (Phase B) — single block, depends on histogram_kernel
//
//  Launched with grid=(1, 1, 1), block=(BLOCK_DIM=256, 1, 1) on the same
//  CUDA stream as histogram_kernel.  Stream ordering gives an implicit
//  grid-wide barrier between Phase A and Phase B with no spin-wait.
//
//  Computes:
//    B.1  expert_offsets / seg_starts / real_bases  (sequential, thread 0)
//    B.2  column-wise exclusive prefix sum of block_hist → block_offset
//    B.3  exclusive prefix sum of naept[1..N_recv] → naept[0..N_recv]
// ============================================================================
__global__ __launch_bounds__(BLOCK_DIM)
void prefix_sums_kernel(
    const int*   __restrict__ tokens_per_expert,    // [E]
    int*         __restrict__ block_hist,            // [scatter_blocks * E] in
    int*         __restrict__ block_offset,          // [scatter_blocks * E] out
    int*         __restrict__ expert_offsets,        // [E+1] out
    int*         __restrict__ seg_starts,            // [E] out
    int*         __restrict__ real_bases,            // [E] out
    int*         __restrict__ naept,                 // [N_recv+1] in/out
    int          N_recv,
    int          num_experts,
    int          alignment,
    int          scatter_blocks)
{
    extern __shared__ char smem[];
    int* scan_buf = reinterpret_cast<int*>(smem);

    // --- B.1: Expert offsets (thread 0, O(E) — typically 8..128) ---
    if (threadIdx.x == 0) {
        int padded_cum = 0, real_cum = 0;
        expert_offsets[0] = 0;
        for (int e = 0; e < num_experts; e++) {
            int count = tokens_per_expert[e];
            int padded = (count > 0) ? ((count + alignment - 1) / alignment * alignment) : 0;
            seg_starts[e] = padded_cum;
            real_bases[e] = real_cum;
            padded_cum += padded;
            real_cum += count;
            expert_offsets[e + 1] = padded_cum;
        }
        naept[0] = 0;  // moved out of histogram_kernel — written here for clarity
    }

    // --- B.2: Column-wise exclusive prefix sum of block_hist → block_offset ---
    // G = BLOCK_DIM / num_experts threads per column; each scans a chunk.
    // E ≤ 256 assumed (otherwise G == 0).
    const int G = (num_experts > 0 && num_experts <= BLOCK_DIM)
                  ? (BLOCK_DIM / num_experts) : 1;
    const int col_id = threadIdx.x / G;
    const int local_tid = threadIdx.x % G;

    if (col_id < num_experts) {
        const int rows_per_thread = (scatter_blocks + G - 1) / G;
        const int row_start = local_tid * rows_per_thread;
        const int row_end_val = row_start + rows_per_thread;
        const int row_end = row_end_val < scatter_blocks ? row_end_val : scatter_blocks;

        int local_sum = 0;
        for (int b = row_start; b < row_end; b++) {
            local_sum += block_hist[b * num_experts + col_id];
        }

        scan_buf[threadIdx.x] = local_sum;
    } else {
        scan_buf[threadIdx.x] = 0;
    }
    __syncthreads();

    if (col_id < num_experts && local_tid == 0) {
        int acc = 0;
        for (int i = 0; i < G; i++) {
            int val = scan_buf[col_id * G + i];
            scan_buf[col_id * G + i] = acc;
            acc += val;
        }
    }
    __syncthreads();

    if (col_id < num_experts) {
        const int rows_per_thread = (scatter_blocks + G - 1) / G;
        const int row_start = local_tid * rows_per_thread;
        const int row_end_val = row_start + rows_per_thread;
        const int row_end = row_end_val < scatter_blocks ? row_end_val : scatter_blocks;

        int running = scan_buf[threadIdx.x];
        for (int b = row_start; b < row_end; b++) {
            int val = block_hist[b * num_experts + col_id];
            block_offset[b * num_experts + col_id] = running;
            running += val;
        }
    }
    __syncthreads();  // ensure scan_buf reusable for B.3

    // --- B.3: naept exclusive prefix sum (one block, BLOCK_DIM threads) ---
    {
        const int chunk = (N_recv + BLOCK_DIM - 1) / BLOCK_DIM;
        const int start_idx = threadIdx.x * chunk;
        const int end_idx_raw = start_idx + chunk;
        const int end_idx = end_idx_raw < N_recv ? end_idx_raw : N_recv;

        int partial_sum = 0;
        for (int i = start_idx; i < end_idx; i++) {
            partial_sum += naept[i + 1];
        }
        scan_buf[threadIdx.x] = partial_sum;
        __syncthreads();

        if (threadIdx.x == 0) {
            int acc = 0;
            for (int i = 0; i < BLOCK_DIM; i++) {
                int val = scan_buf[i];
                scan_buf[i] = acc;
                acc += val;
            }
            naept[N_recv] = acc;
        }
        __syncthreads();

        int running = scan_buf[threadIdx.x];
        for (int i = start_idx; i < end_idx; i++) {
            int val = naept[i + 1];
            naept[i] = running;
            running += val;
        }
    }
}

// ============================================================================
//  Kernel 2: Scatter + Fixup (fully parallel, ZERO synchronization)
//
//  Each block processes ROWS_PER_BLOCK=32 token rows for scatter,
//  then processes a chunk of TK_padded for fixup.
//  All data needed (block_offset, expert_offsets, naept) is precomputed.
// ============================================================================
template <int TOPK>
__global__ __launch_bounds__(BLOCK_DIM)
void scatter_and_fixup_kernel(
    const int*   __restrict__ dispatched_indices,   // [N_recv, topk]
    const float* __restrict__ dispatched_probs,     // [N_recv, topk]
    const int*   __restrict__ block_offset,          // [scatter_blocks * E]
    const int*   __restrict__ expert_offsets,        // [E+1] (padded cumsum)
    const int*   __restrict__ seg_starts,            // [E]
    const int*   __restrict__ tokens_per_expert,    // [E]
    const int*   __restrict__ naept,                 // [N_recv+1]
    int*         __restrict__ x_gather_idx,          // [TK_padded] output
    int*         __restrict__ s_scatter_idx,         // [TK_padded] output
    int*         __restrict__ s_reverse_scatter_idx, // [TK] output
    float*       __restrict__ topk_scores,           // [TK_padded] output
    int          N_recv,
    int          num_experts,
    int          topk_param,
    int          TK,
    int          TK_padded,
    int          scatter_blocks)
{
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    constexpr int warp_num = BLOCK_DIM >> 5;

    // ═══════════ Phase 1: Scatter (warp-ballot, deterministic ordering) ═══════
    if (blockIdx.x < scatter_blocks) {
        extern __shared__ char smem[];
        uint32_t* expert_bitmask = reinterpret_cast<uint32_t*>(smem);
        // Also cache block_offset for this block
        int* my_offset = reinterpret_cast<int*>(smem + num_experts * sizeof(uint32_t));

        // Load this block's precomputed offsets + init bitmask
        for (int i = threadIdx.x; i < num_experts; i += BLOCK_DIM) {
            expert_bitmask[i] = 0u;
            my_offset[i] = block_offset[blockIdx.x * num_experts + i];
        }
        __syncthreads();

        const int block_row_base = blockIdx.x * ROWS_PER_BLOCK;
        const int global_row = block_row_base + lane_id;
        const bool row_valid = global_row < N_recv;

        // Load topk entries + build bitmask
        int reg_expert[MAX_TOPK];
        float reg_prob[MAX_TOPK];

        #pragma unroll
        for (int k = 0; k < TOPK; k++) {
            reg_expert[k] = -1;
            reg_prob[k] = 0.0f;
        }

        #pragma unroll
        for (int col = 0; col < TOPK; col++) {
            if (col >= topk_param) break;
            int expert = -1;
            float prob = 0.0f;
            if (row_valid) {
                expert = dispatched_indices[global_row * topk_param + col];
                prob = dispatched_probs[global_row * topk_param + col];
            }
            if (expert >= 0 && expert < num_experts) {
                if (col % warp_num == warp_id) {
                    atomicOr(&expert_bitmask[expert], 1u << lane_id);
                }
                reg_expert[col] = expert;
                reg_prob[col] = prob;
            }
        }
        __syncthreads();

        // Assign positions using ballot prefix count (deterministic, stable)
        int reg_padded_pos[MAX_TOPK];
        #pragma unroll
        for (int k = 0; k < TOPK; k++) reg_padded_pos[k] = -1;

        for (int expert_id = warp_id; expert_id < num_experts; expert_id += warp_num) {
            const uint32_t mask = expert_bitmask[expert_id];
            if (mask == 0u) continue;

            // This block's starting offset for this expert (precomputed, no spin!)
            const int base_offset = my_offset[expert_id];
            const bool lane_active = (mask & (1u << lane_id)) != 0;

            if (lane_active && row_valid) {
                // Intra-block position: stable via ballot prefix count
                int intra_pos = base_offset + __popc(mask & ((1u << lane_id) - 1));
                int padded_pos = seg_starts[expert_id] + intra_pos;

                // Find which topk slot matches this expert
                #pragma unroll
                for (int k = 0; k < TOPK; k++) {
                    if (reg_expert[k] == expert_id) {
                        reg_padded_pos[k] = padded_pos;
                        break;
                    }
                }
            }
        }

        // Write all outputs for this token
        // BUG FIX: within_token_rank must be computed globally across ALL warps.
        // Each warp only sets reg_padded_pos for its assigned experts, but all
        // threads have the complete reg_expert array (all topk cols were read).
        // Compute rank = count of valid expert slots with smaller expert_id
        // to give stable ascending-expert ordering within each token.
        if (row_valid) {
            const int naept_base = naept[global_row];

            #pragma unroll
            for (int k = 0; k < TOPK; k++) {
                if (k >= topk_param) break;
                if (reg_padded_pos[k] >= 0) {
                    const int padded_pos = reg_padded_pos[k];

                    // Compute global rank: how many of this token's valid
                    // expert assignments have expert_id < reg_expert[k]?
                    int rank = 0;
                    #pragma unroll
                    for (int j = 0; j < TOPK; j++) {
                        if (j >= topk_param) break;
                        if (reg_expert[j] >= 0 && reg_expert[j] < reg_expert[k]) {
                            rank++;
                        }
                    }

                    const int token_major_pos = naept_base + rank;

                    // Write ALL outputs (no conflicts — positions are unique)
                    x_gather_idx[padded_pos] = global_row;
                    s_scatter_idx[padded_pos] = token_major_pos;
                    s_reverse_scatter_idx[token_major_pos] = padded_pos;
                    topk_scores[token_major_pos] = reg_prob[k];
                }
            }
        }
    }

    // ═══════════ Phase 2: Pad-fill (vectorized, coalesced) ════════════════════
    // Fill padding positions: x_gather_idx=0, s_scatter_idx=TK, topk_scores=0
    // Process all TK_padded positions, skip real ones.
    // Use grid-stride loop for full coverage.

    const int total_threads = gridDim.x * BLOCK_DIM;
    const int global_tid = blockIdx.x * BLOCK_DIM + threadIdx.x;

    for (int pos = global_tid; pos < TK_padded; pos += total_threads) {
        // Binary search for expert
        int lo = 0, hi = num_experts;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (expert_offsets[mid + 1] <= pos) lo = mid + 1;
            else hi = mid;
        }
        const int seg_start = expert_offsets[lo];
        const int real_count = tokens_per_expert[lo];
        const int local_pos = pos - seg_start;

        if (local_pos >= real_count) {
            // Padding position: fill defaults
            x_gather_idx[pos] = 0;
            s_scatter_idx[pos] = TK;  // points to topk_scores[TK]=0
            // topk_scores[pos] already 0 from zero-init
        }
    }
}

// ============================================================================
//  C++ entry point
// ============================================================================
void deepep_topk_metadata_cuda(
    torch::Tensor& dispatched_indices,
    torch::Tensor& dispatched_probs,
    torch::Tensor& tokens_per_expert,
    torch::Tensor& expert_offsets,
    torch::Tensor& seg_starts,
    torch::Tensor& real_bases,
    torch::Tensor& x_gather_idx,
    torch::Tensor& s_scatter_idx,
    torch::Tensor& s_reverse_scatter_idx,
    torch::Tensor& topk_scores,
    torch::Tensor& naept,
    torch::Tensor& global_block_cumsum,  // reused as workspace
    int64_t N_recv,
    int64_t E,
    int64_t topk,
    int64_t TK,
    int64_t TK_padded,
    int64_t alignment,
    int64_t stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    if (N_recv == 0 || TK == 0) {
        cudaMemsetAsync(naept.data_ptr<int>(), 0, (N_recv + 1) * sizeof(int), stream);
        return;
    }

    const int scatter_blocks = (N_recv + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    // Workspace layout within global_block_cumsum:
    //   [0 .. scatter_blocks*E-1]:                  block_hist
    //   [scatter_blocks*E .. 2*scatter_blocks*E-1]: block_offset
    //   [2*scatter_blocks*E] (legacy):              previously completion_flag
    //                                               (no longer used; barrier
    //                                                is now a separate kernel
    //                                                launch).
    int* workspace = global_block_cumsum.data_ptr<int>();
    int* block_hist = workspace;
    int* block_offset = workspace + scatter_blocks * E;

    // Shared memory for histogram_kernel: expert_bitmask[E]
    int smem_hist = static_cast<int>(E * sizeof(uint32_t));
    // Shared memory for prefix_sums_kernel: scan_buf[BLOCK_DIM]
    int smem_prefix = static_cast<int>(BLOCK_DIM * sizeof(int));

    dim3 grid1(scatter_blocks);
    dim3 block1(BLOCK_DIM);

    // ── Kernel 1a: Phase A — per-block histogram (no inter-block sync) ───────
    #define LAUNCH_HIST(TV) \
        histogram_kernel<TV><<<grid1, block1, smem_hist, stream>>>( \
            dispatched_indices.data_ptr<int>(), \
            block_hist, \
            naept.data_ptr<int>(), \
            static_cast<int>(N_recv), static_cast<int>(E), \
            static_cast<int>(topk));

    if (topk <= 4) { LAUNCH_HIST(4); }
    else if (topk <= 8) { LAUNCH_HIST(8); }
    else { LAUNCH_HIST(16); }
    #undef LAUNCH_HIST

    // ── Kernel 1b: Phase B — prefix sums (single block, runs after 1a) ───────
    // Stream ordering between 1a and 1b provides the inter-phase grid barrier
    // that the legacy combined kernel implemented with an ad-hoc spin-wait.
    prefix_sums_kernel<<<dim3(1), block1, smem_prefix, stream>>>(
        tokens_per_expert.data_ptr<int>(),
        block_hist, block_offset,
        expert_offsets.data_ptr<int>(),
        seg_starts.data_ptr<int>(),
        real_bases.data_ptr<int>(),
        naept.data_ptr<int>(),
        static_cast<int>(N_recv), static_cast<int>(E),
        static_cast<int>(alignment), scatter_blocks);

    // ── Kernel 2: Scatter + fixup ────────────────────────────────────────────
    // Grid must cover ALL scatter blocks: scatter phase relies on blockIdx.x
    // mapping 1:1 to a 32-row chunk of N_recv (block_row_base = blockIdx.x*32).
    // Capping the grid here would silently skip rows beyond cap*32 — which is
    // exactly what caused the SEQ_LEN ≥ 12K illegal-memory-access in
    // token_gather_sum_kernel: scatter_blocks=2712 capped at 2048 left rows
    // 65536..86768 unscattered, so x_gather_idx beyond TK_padded position
    // 2048*32 stayed at the pad-fill default and downstream gathers tried to
    // read 0-row data using s_scatter_idx pointing past valid topk_scores.
    // Pad-fill (Phase 2) uses a grid-stride loop, so any grid ≥ scatter_blocks
    // is correct and equally efficient.
    int grid2_blocks = scatter_blocks;
    int padfill_blocks = (int)(TK_padded + BLOCK_DIM - 1) / BLOCK_DIM;
    if (padfill_blocks > grid2_blocks) grid2_blocks = padfill_blocks;

    // Shared memory: expert_bitmask[E] + my_offset[E]
    int smem_k2 = (E * sizeof(uint32_t)) + (E * sizeof(int));

    dim3 grid2(grid2_blocks);
    dim3 block2(BLOCK_DIM);

    #define LAUNCH_K2(TV) \
        scatter_and_fixup_kernel<TV><<<grid2, block2, smem_k2, stream>>>( \
            dispatched_indices.data_ptr<int>(), \
            dispatched_probs.data_ptr<float>(), \
            block_offset, \
            expert_offsets.data_ptr<int>(), \
            seg_starts.data_ptr<int>(), \
            tokens_per_expert.data_ptr<int>(), \
            naept.data_ptr<int>(), \
            x_gather_idx.data_ptr<int>(), \
            s_scatter_idx.data_ptr<int>(), \
            s_reverse_scatter_idx.data_ptr<int>(), \
            topk_scores.data_ptr<float>(), \
            static_cast<int>(N_recv), static_cast<int>(E), \
            static_cast<int>(topk), static_cast<int>(TK), \
            static_cast<int>(TK_padded), scatter_blocks);

    if (topk <= 4) { LAUNCH_K2(4); }
    else if (topk <= 8) { LAUNCH_K2(8); }
    else { LAUNCH_K2(16); }
    #undef LAUNCH_K2
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("deepep_topk_metadata_cuda",
          &deepep_topk_metadata_cuda,
          "DeepEP topk metadata: 2-kernel histogram + scatter (CUDA)",
          py::arg("dispatched_indices"),
          py::arg("dispatched_probs"),
          py::arg("tokens_per_expert"),
          py::arg("expert_offsets"),
          py::arg("seg_starts"),
          py::arg("real_bases"),
          py::arg("x_gather_idx"),
          py::arg("s_scatter_idx"),
          py::arg("s_reverse_scatter_idx"),
          py::arg("topk_scores"),
          py::arg("naept"),
          py::arg("global_block_cumsum"),
          py::arg("N_recv"),
          py::arg("E"),
          py::arg("topk"),
          py::arg("TK"),
          py::arg("TK_padded"),
          py::arg("alignment"),
          py::arg("stream"));
}
