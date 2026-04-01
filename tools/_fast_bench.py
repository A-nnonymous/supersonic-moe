"""Fast focused benchmark: key shapes only, unbuffered output."""
import gc, os, sys
os.environ["USE_QUACK_GEMM"] = "1"
for _k in ["PADDLE_ELASTIC_JOB_ID","PADDLE_TRAINER_ENDPOINTS","DISTRIBUTED_TRAINER_ENDPOINTS","FLAGS_START_PORT","PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"
os.environ["PADDLE_TRAINERS_NUM"] = "1"
for _k in ["SONIC_MOE_FP8_MODE","SONIC_MOE_FP8_ASSUME_ALIGNED","SONIC_MOE_FP8_FUSED_SWIGLU_QUANT","SONIC_MOE_FP8_SAVE_Z_FP8","SONIC_MOE_FP8_FUSED_GATED","SONIC_MOE_FP8_WGRAD"]:
    os.environ.pop(_k, None)

import torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")

SHAPES = [
    dict(name="S1", T=4096, H=4096, I=1024,  E=128, K=8, desc="small-I"),
    dict(name="S2", T=4096, H=4096, I=2048,  E=128, K=8, desc="medium-I"),
    dict(name="S4", T=8192, H=3072, I=1536,  E=8,   K=8, desc="production"),
]
WARMUP, BENCH_ITERS, SEEDS = 3, 6, [42, 123, 777]

def build_uniform(T,K,E,dev):
    tok=torch.arange(T,device=dev).unsqueeze(1); off=torch.arange(K,device=dev).unsqueeze(0)
    return torch.full((T,K),1./K,dtype=torch.float32,device=dev), ((tok*K+off)%E).to(torch.int32)

def reset():
    import sonicmoe.functional as F
    from sonicmoe.functional import clear_all_fp8_weight_caches
    clear_all_fp8_weight_caches(); F._ALIGNMENT_STREAK=0; F._ALIGNMENT_ASSUMED=False

def rrmse(a,b): return((a.float()-b.float()).norm()/b.float().norm()).item()*100
def corr(a,b):
    af,bf=a.float().flatten(),b.float().flatten(); af,bf=af-af.mean(),bf-bf.mean()
    return(af@bf/(af.norm()*bf.norm())).item()

def measure(cfg, use_fp8):
    from sonicmoe import MoE, enable_quack_gemm; from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F
    T,H,I,E,K=cfg["T"],cfg["H"],cfg["I"],cfg["E"],cfg["K"]
    reset(); torch.manual_seed(42)
    moe=MoE(num_experts=E,num_experts_per_tok=K,hidden_size=H,intermediate_size=I,
            activation_function=ActivationType.SWIGLU,add_bias=False,std=0.02).to("cuda",torch.bfloat16)
    x_base=torch.randn(T,H,device="cuda",dtype=torch.bfloat16); grad=torch.randn_like(x_base)
    sc,idx=build_uniform(T,K,E,torch.device("cuda"))
    class UR(torch.autograd.Function):
        @staticmethod
        def forward(ctx,rl,Ea,Ka): ctx.save_for_backward(sc,idx); ctx.E=Ea; ctx.d=rl.dtype; return sc.clone(),idx.clone()
        @staticmethod
        def backward(ctx,gs,_): s,_=ctx.saved_tensors; return torch.zeros(s.size(0),ctx.E,dtype=ctx.d,device=s.device),None,None
    orig=F.TC_Softmax_Topk_Router_Function; F.TC_Softmax_Topk_Router_Function=UR
    try:
        enable_quack_gemm(); moe.train()
        for _ in range(WARMUP):
            moe.zero_grad(set_to_none=True); x=x_base.clone().requires_grad_(True)
            o,_=moe(x,use_fp8=use_fp8); o.backward(grad)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        s_e=torch.cuda.Event(enable_timing=True); e_e=torch.cuda.Event(enable_timing=True); s_e.record()
        for _ in range(BENCH_ITERS):
            moe.zero_grad(set_to_none=True); x=x_base.clone().requires_grad_(True)
            o,_=moe(x,use_fp8=use_fp8); o.backward(grad)
        e_e.record(); torch.cuda.synchronize()
        ms=s_e.elapsed_time(e_e)/BENCH_ITERS; gb=torch.cuda.max_memory_allocated()/(1024**3)
    finally: F.TC_Softmax_Topk_Router_Function=orig
    del moe,x_base,grad; reset(); torch.cuda.empty_cache(); gc.collect()
    return ms, gb

def precision(cfg):
    from sonicmoe import MoE, enable_quack_gemm; from sonicmoe.enums import ActivationType
    import sonicmoe.functional as F
    T,H,I,E,K=cfg["T"],cfg["H"],cfg["I"],cfg["E"],cfg["K"]
    results=[]
    for seed in SEEDS:
        reset(); torch.manual_seed(seed)
        moe=MoE(num_experts=E,num_experts_per_tok=K,hidden_size=H,intermediate_size=I,
                activation_function=ActivationType.SWIGLU,add_bias=False,std=0.02).to("cuda",torch.bfloat16)
        x_base=torch.randn(T,H,device="cuda",dtype=torch.bfloat16); grad=torch.randn_like(x_base)
        sc,idx=build_uniform(T,K,E,torch.device("cuda"))
        class UR(torch.autograd.Function):
            @staticmethod
            def forward(ctx,rl,Ea,Ka): ctx.save_for_backward(sc,idx); ctx.E=Ea; ctx.d=rl.dtype; return sc.clone(),idx.clone()
            @staticmethod
            def backward(ctx,gs,_): s,_=ctx.saved_tensors; return torch.zeros(s.size(0),ctx.E,dtype=ctx.d,device=s.device),None,None
        orig=F.TC_Softmax_Topk_Router_Function; F.TC_Softmax_Topk_Router_Function=UR
        try:
            enable_quack_gemm(); moe.train()
            reset(); moe.zero_grad(set_to_none=True); x=x_base.clone().requires_grad_(True)
            o16,_=moe(x); o16.backward(grad); dx16=x.grad.clone(); fwd16=o16.detach()
            reset(); moe.zero_grad(set_to_none=True); x=x_base.clone().requires_grad_(True)
            o8,_=moe(x,use_fp8=True); o8.backward(grad); dx8=x.grad.clone(); fwd8=o8.detach()
            fr,fc=rrmse(fwd8,fwd16),corr(fwd8,fwd16); br,bc=rrmse(dx8,dx16),corr(dx8,dx16)
            ok=fr<10 and fc>0.99 and br<10 and bc>0.99
            results.append(dict(seed=seed,fr=fr,fc=fc,br=br,bc=bc,ok=ok))
        finally: F.TC_Softmax_Topk_Router_Function=orig
        del moe,x_base,grad; reset(); torch.cuda.empty_cache(); gc.collect()
    return results

if __name__=="__main__":
    print("="*80); print(f"BENCHMARK: SonicMoE BF16 vs FP8 | GPU: {torch.cuda.get_device_name()}"); print("="*80)
    all_res={}
    for cfg in SHAPES:
        sn=cfg["name"]; T,H,I,E,K=cfg["T"],cfg["H"],cfg["I"],cfg["E"],cfg["K"]
        tpe=T*K//E; print(f"\n--- {sn}: T={T},H={H},I={I},E={E},K={K} tpe={tpe} [{cfg['desc']}] ---")
        bf16_ms,bf16_gb=measure(cfg,False); print(f"  BF16: {bf16_ms:.2f}ms {bf16_gb:.2f}GiB")
        fp8_ms,fp8_gb=measure(cfg,True);   sp=bf16_ms/fp8_ms; print(f"  FP8:  {fp8_ms:.2f}ms {fp8_gb:.2f}GiB ({sp:.2f}x)")
        prec=precision(cfg); aok=all(p["ok"] for p in prec)
        for p in prec:
            s="✓" if p["ok"] else "✗"
            print(f"    seed={p['seed']}: fwd={p['fr']:.2f}%/{p['fc']:.4f} bwd={p['br']:.2f}%/{p['bc']:.4f} {s}")
        all_res[sn]=dict(bf16_ms=bf16_ms,fp8_ms=fp8_ms,bf16_gb=bf16_gb,fp8_gb=fp8_gb,speedup=sp,prec_ok=aok,desc=cfg["desc"])
    print(f"\n{'='*80}\nSUMMARY")
    print(f"{'Shape':<6} {'Desc':<14} {'BF16(ms)':>9} {'FP8(ms)':>8} {'Speed':>6} {'BF16(G)':>8} {'FP8(G)':>7} {'Prec':>5}")
    print("-"*65)
    for sn,r in all_res.items():
        ps="✓" if r["prec_ok"] else "✗"
        print(f"{sn:<6} {r['desc']:<14} {r['bf16_ms']:9.2f} {r['fp8_ms']:8.2f} {r['speedup']:5.2f}x {r['bf16_gb']:8.2f} {r['fp8_gb']:7.2f} {ps:>5}")
    print("="*80)
