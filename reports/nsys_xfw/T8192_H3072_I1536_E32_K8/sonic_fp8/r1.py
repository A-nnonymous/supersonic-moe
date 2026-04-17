
import gc, json, os, sys
import numpy as np
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

import torch, torch.nn.functional as F

data_dir, S, H, I, E, K = "/tmp/moe_xfw_raa7xzha/E32/seed42", 8192, 3072, 1536, 32, 8
mode = "nsys"
warmup, iters = 5, 20
device = torch.device("cuda:0")
use_fp8 = True

def split_to_interleaved(w):
    h = w.shape[0] // 2
    o = torch.empty_like(w); o[0::2] = w[:h]; o[1::2] = w[h:]
    return o

if mode == "precision":
    # ── Precision: use _UpProjection + _DownProjection with fixed routing ──
    from sonicmoe.functional import _UpProjection, _DownProjection, clear_all_fp8_weight_caches
    from sonicmoe.functional import _refresh_fp8_config
    from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    import sonicmoe.functional as functional

    clear_all_fp8_weight_caches()

    x = torch.from_numpy(np.load(os.path.join(data_dir, "x.npy"))).to(device=device, dtype=torch.bfloat16)
    topk_indices = torch.from_numpy(np.load(os.path.join(data_dir, "topk_indices.npy"))).to(device=device)
    topk_scores = torch.from_numpy(np.load(os.path.join(data_dir, "topk_scores.npy"))).to(device=device)

    w1l, w2l = [], []
    for e in range(E):
        w1_e = torch.from_numpy(np.load(os.path.join(data_dir, f"w1_e{e}.npy")))
        w2_e = torch.from_numpy(np.load(os.path.join(data_dir, f"w2_e{e}.npy")))
        w1l.append(split_to_interleaved(w1_e.T)); w2l.append(w2_e.T)
    w1_param = torch.stack(w1l).to(device=device, dtype=torch.bfloat16).contiguous()
    w2_param = torch.stack(w2l).to(device=device, dtype=torch.bfloat16).contiguous()
    w1_param.requires_grad_(True)
    w2_param.requires_grad_(True)
    del w1l, w2l

    TK = S * K
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    expert_frequency = torch.empty(E, dtype=torch.int32, device=device)
    expert_frequency_offset = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    TC_topk_router_metadata_triton(
        topk_indices, E, expert_frequency, expert_frequency_offset,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )

    def run_sonic_fwd():
        w1f = w1_param.permute(1, 2, 0)
        w2f = w2_param.permute(1, 2, 0)
        if use_fp8:
            with enable_fp8(True):
                _refresh_fp8_config()
                try:
                    y1, z = _UpProjection.apply(
                        x, w1f, None, expert_frequency_offset, TK, K, 0,
                        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                        None, False, ActivationType.SWIGLU, False, False,
                    )
                    o = _DownProjection.apply(
                        y1, z, w2f, None, topk_scores, topk_indices,
                        expert_frequency_offset, S, K, 0, x_gather_idx,
                        s_scatter_idx, s_reverse_scatter_idx, None, False,
                        ActivationType.SWIGLU, None,
                    )
                finally:
                    clear_all_fp8_weight_caches()
        else:
            with enable_fp8(False):
                y1, z = _UpProjection.apply(
                    x, w1f, None, expert_frequency_offset, TK, K, 0,
                    x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
                    None, False, ActivationType.SWIGLU, False, False,
                )
                o = _DownProjection.apply(
                    y1, z, w2f, None, topk_scores, topk_indices,
                    expert_frequency_offset, S, K, 0, x_gather_idx,
                    s_scatter_idx, s_reverse_scatter_idx, None, False,
                    ActivationType.SWIGLU, None,
                )
        return o

    for _ in range(2):
        _ = run_sonic_fwd()
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Measured fwd + bwd
    x_input = x.detach().clone().requires_grad_(True)
    # Temporarily swap x for x_input to capture dx
    _orig_x = x
    x = x_input
    o = run_sonic_fwd()
    grad_out = torch.from_numpy(np.load(os.path.join(data_dir, "grad_output.npy"))).to(device=device, dtype=torch.bfloat16)
    o.backward(grad_out)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    x = _orig_x
    clear_all_fp8_weight_caches()
    np.save(os.path.join(data_dir, "sonic_fp8_output.npy"), o.detach().float().cpu().numpy())
    # Save gradients
    pref = "sonic_fp8_output.npy".replace("_output.npy", "")
    if x_input.grad is not None:
        np.save(os.path.join(data_dir, f"{pref}_dx.npy"), x_input.grad.float().cpu().numpy())
    if w1_param.grad is not None:
        np.save(os.path.join(data_dir, f"{pref}_dw1.npy"), w1_param.grad.float().cpu().numpy())
    if w2_param.grad is not None:
        np.save(os.path.join(data_dir, f"{pref}_dw2.npy"), w2_param.grad.float().cpu().numpy())
    print(json.dumps({"peak_mem_mib": round(peak_mem, 1), "status": "ok"}))

elif mode == "nsys":
    # ── nsys: matches introspect.py frontier path exactly ──
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    import sonicmoe.functional as functional

    functional.clear_all_fp8_weight_caches()
    functional._ALIGNMENT_ASSUMED = True

    moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
               intermediate_size=I, activation_function=ActivationType.SWIGLU,
               add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)

    if use_fp8:
        moe.refresh_fp8_shadow_weights()
        moe.stash_bf16_to_cpu()

    x = torch.from_numpy(np.load(os.path.join(data_dir, "x.npy"))).to(device=device, dtype=torch.bfloat16)

    use_token_rounding = (E > 8)
    w1_p = moe.c_fc.weight.permute(1, 2, 0)
    w2_p = moe.c_proj.weight.permute(1, 2, 0)

    if use_token_rounding:
        # E>8: token rounding + moe_general_routing_inputs (same as introspect.py)
        from sonicmoe.functional import count_cumsum, moe_general_routing_inputs

        def run_iter():
            Mtile = 128
            xw = x.detach().clone().requires_grad_(True)
            with torch.no_grad():
                rl = F.linear(xw, moe.router.weight)
                sc = F.softmax(rl, dim=-1, dtype=torch.float32).to(torch.bfloat16)
                tv, ti = sc.topk(K, dim=-1)
                tv /= tv.sum(dim=-1, keepdim=True)
                sc.scatter_(-1, ti, tv)
                cb = sc.clone() - 1; cb.scatter_(1, ti, tv)
                si = cb.argsort(dim=0, descending=True).int()
                ef = count_cumsum(ti.view(-1), E, do_cumsum=True)[0]
                efr = (torch.ceil(ef / Mtile) * Mtile).int()
                mk = torch.arange(S, device=device, dtype=torch.int32)[:, None].expand(-1, E) < efr[None, :]
                tok = si[mk]; exp = torch.arange(E, device=device, dtype=torch.int32)[None, :].expand(S, -1)[mk]
                od = tok.argsort().int(); tok = tok[od]; exp = exp[od]
                rsc = sc[tok, exp].contiguous()
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        out, _ = moe_general_routing_inputs(
                            xw, rsc, tok, exp, w1_p, None, w2_p, None,
                            E, moe.stream_id, ActivationType.SWIGLU, False)
                else:
                    with enable_fp8(False):
                        out, _ = moe_general_routing_inputs(
                            xw, rsc, tok, exp, w1_p, None, w2_p, None,
                            E, moe.stream_id, ActivationType.SWIGLU, False)
            return xw, out
    else:
        # E<=8: direct moe(xw)
        def run_iter():
            xw = x.detach().clone().requires_grad_(True)
            with enable_quack_gemm(True):
                if use_fp8:
                    with enable_fp8(True):
                        o, aux = moe(xw, use_fp8=True)
                else:
                    with enable_fp8(False):
                        o, aux = moe(xw)
            return xw, o

    for _ in range(warmup):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True)
        del xw, o
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        xw, o = run_iter()
        o.sum().backward()
        moe.zero_grad(set_to_none=True)
        del xw, o
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("NSYS_DONE", flush=True)
