"""Diagnose: compare old env-var approach vs new use_fp8 API."""
import gc, os, sys, time
os.environ["USE_QUACK_GEMM"] = "1"
for _k in ["PADDLE_ELASTIC_JOB_ID","PADDLE_TRAINER_ENDPOINTS","DISTRIBUTED_TRAINER_ENDPOINTS","FLAGS_START_PORT","PADDLE_ELASTIC_TIMEOUT"]:
    os.environ.pop(_k, None)
os.environ["NNODES"] = "1"; os.environ["PADDLE_TRAINERS_NUM"] = "1"

import torch
sys.path.insert(0, "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe")
import sonicmoe.functional as F_mod
from sonicmoe import MoE, enable_quack_gemm
from sonicmoe.enums import ActivationType
from sonicmoe.functional import clear_all_fp8_weight_caches

T,H,I,E,K = 4096, 4096, 1024, 128, 8
WARMUP, ITERS = 5, 10

def build_uniform(T,K,E,dev):
    tok=torch.arange(T,device=dev).unsqueeze(1); off=torch.arange(K,device=dev).unsqueeze(0)
    return torch.full((T,K),1./K,dtype=torch.float32,device=dev), ((tok*K+off)%E).to(torch.int32)

sc,idx = build_uniform(T,K,E,torch.device("cuda"))
class UR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rl,Ea,Ka): ctx.save_for_backward(sc,idx); ctx.E=Ea; ctx.d=rl.dtype; return sc.clone(),idx.clone()
    @staticmethod
    def backward(ctx,gs,_): s,_=ctx.saved_tensors; return torch.zeros(s.size(0),ctx.E,dtype=ctx.d,device=s.device),None,None

def setup_and_measure(label, setup_fn):
    setup_fn()
    torch.manual_seed(42)
    moe = MoE(num_experts=E,num_experts_per_tok=K,hidden_size=H,intermediate_size=I,
              activation_function=ActivationType.SWIGLU,add_bias=False,std=0.02).to("cuda",torch.bfloat16)
    enable_quack_gemm()
    x_base = torch.randn(T,H,device="cuda",dtype=torch.bfloat16); grad = torch.randn_like(x_base)
    orig = F_mod.TC_Softmax_Topk_Router_Function; F_mod.TC_Softmax_Topk_Router_Function = UR
    moe.train()
    for _ in range(WARMUP):
        moe.zero_grad(set_to_none=True); x=x_base.clone().requires_grad_(True)
        o,_=moe(x); o.backward(grad)
    torch.cuda.synchronize()
    # Check alignment state
    print(f"  [{label}] _ALIGNMENT_ASSUMED={F_mod._ALIGNMENT_ASSUMED}, _ALIGNMENT_STREAK={F_mod._ALIGNMENT_STREAK}")
    print(f"  [{label}] _fp8_mode={F_mod._fp8_mode()}, _fp8_enabled={F_mod._fp8_enabled()}")
    print(f"  [{label}] _fp8_lean={F_mod._fp8_lean()}, _fused_gated={F_mod._use_fused_blockscaled_gated()}")

    torch.cuda.reset_peak_memory_stats()
    se=torch.cuda.Event(enable_timing=True); ee=torch.cuda.Event(enable_timing=True); se.record()
    for _ in range(ITERS):
        moe.zero_grad(set_to_none=True); x=x_base.clone().requires_grad_(True)
        o,_=moe(x); o.backward(grad)
    ee.record(); torch.cuda.synchronize()
    ms=se.elapsed_time(ee)/ITERS; gb=torch.cuda.max_memory_allocated()/(1024**3)
    print(f"  [{label}] {ms:.2f}ms, {gb:.2f}GiB")
    F_mod.TC_Softmax_Topk_Router_Function = orig
    del moe,x_base,grad; torch.cuda.empty_cache(); gc.collect()
    return ms, gb

# 1. BF16 (no FP8)
def setup_bf16():
    for k in ["SONIC_MOE_FP8_MODE","SONIC_MOE_FP8_FUSED_GATED","SONIC_MOE_FP8_WGRAD","SONIC_MOE_FP8_ASSUME_ALIGNED","SONIC_MOE_FP8_FUSED_SWIGLU_QUANT","SONIC_MOE_FP8_SAVE_Z_FP8"]:
        os.environ.pop(k, None)
    import sonicmoe.functional.utils as u; u._IS_FP8_ACTIVE = False
    clear_all_fp8_weight_caches(); F_mod._ALIGNMENT_ASSUMED=False; F_mod._ALIGNMENT_STREAK=0
print("=== BF16 (baseline) ===")
bf16_ms, bf16_gb = setup_and_measure("BF16", setup_bf16)

# 2. FP8 via env vars (old approach)
def setup_fp8_env():
    os.environ["SONIC_MOE_FP8_MODE"] = "perf"
    os.environ["SONIC_MOE_FP8_FUSED_GATED"] = "1"
    os.environ["SONIC_MOE_FP8_WGRAD"] = "0"
    os.environ["SONIC_MOE_FP8_ASSUME_ALIGNED"] = "1"
    os.environ["SONIC_MOE_FP8_FUSED_SWIGLU_QUANT"] = "1"
    os.environ["SONIC_MOE_FP8_SAVE_Z_FP8"] = "1"
    import sonicmoe.functional.utils as u; u._IS_FP8_ACTIVE = False
    clear_all_fp8_weight_caches(); F_mod._ALIGNMENT_ASSUMED=True; F_mod._ALIGNMENT_STREAK=3
print("\n=== FP8 via ENV VARS (old approach) ===")
fp8env_ms, fp8env_gb = setup_and_measure("FP8-env", setup_fp8_env)

# 3. FP8 via use_fp8=True (new approach) - but that's in moe.forward()
# For this diag we simulate it by setting the flag directly
def setup_fp8_api():
    for k in ["SONIC_MOE_FP8_MODE","SONIC_MOE_FP8_FUSED_GATED","SONIC_MOE_FP8_WGRAD","SONIC_MOE_FP8_ASSUME_ALIGNED","SONIC_MOE_FP8_FUSED_SWIGLU_QUANT","SONIC_MOE_FP8_SAVE_Z_FP8"]:
        os.environ.pop(k, None)
    import sonicmoe.functional.utils as u; u._IS_FP8_ACTIVE = True
    clear_all_fp8_weight_caches(); F_mod._ALIGNMENT_ASSUMED=False; F_mod._ALIGNMENT_STREAK=0
print("\n=== FP8 via is_fp8_active (new API) ===")
fp8api_ms, fp8api_gb = setup_and_measure("FP8-api", setup_fp8_api)

print(f"\n=== COMPARISON ===")
print(f"BF16:     {bf16_ms:.2f}ms  {bf16_gb:.2f}GiB")
print(f"FP8-env:  {fp8env_ms:.2f}ms  {fp8env_gb:.2f}GiB  ({bf16_ms/fp8env_ms:.2f}x)")
print(f"FP8-api:  {fp8api_ms:.2f}ms  {fp8api_gb:.2f}GiB  ({bf16_ms/fp8api_ms:.2f}x)")
