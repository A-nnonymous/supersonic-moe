# Monkey-patch quack to support FP8 (float8_e4m3fn) tensors.
# The upstream quack package lacks FP8 in its dtype map and doesn't handle
# FP8 DLPack conversion.  GemmSm100 already validates FP8 as a legal
# input dtype, so only the utility layer needs patching.

_applied = False


def apply_fp8_quack_patch() -> None:
    """Idempotent patch: call at import time before any quack.gemm usage."""
    global _applied
    if _applied:
        return
    _applied = True

    import cutlass
    import torch
    from cutlass.cute.runtime import from_dlpack
    from quack.cute_dsl_utils import torch2cute_dtype_map
    from quack.gemm_wrapper_utils import GemmWrapperBase

    # 1. Extend dtype map
    torch2cute_dtype_map[torch.float8_e4m3fn] = cutlass.Float8E4M3FN

    # 2. Override create_cute_tensor to handle FP8 via uint8 view.
    #    fp8 and uint8 are both 1 byte, so .view(torch.uint8) preserves
    #    shape and strides even on non-contiguous tensors.
    def _fp8_create_cute_tensor(tensor, major, dims, assumed_align=16):
        if tensor is None:
            return None
        leading_dim = 1 if major == dims[1] else 0
        if tensor.dtype == torch.float8_e4m3fn:
            storage = tensor.detach().view(torch.uint8)
            cute_tensor = from_dlpack(storage, assumed_align=assumed_align)
            cute_tensor.element_type = cutlass.Float8E4M3FN
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)
        return from_dlpack(
            tensor.detach(), assumed_align=assumed_align
        ).mark_layout_dynamic(leading_dim=leading_dim)

    GemmWrapperBase.create_cute_tensor = staticmethod(_fp8_create_cute_tensor)

    # 3. Fix CUTLASS DSL alignment bug: varlen + colvec_scale.
    #    QuACK's gemm_default_epi.py line 127-129 uses domain_offset with a
    #    dynamic cu_seqlens_m[batch_idx] on a bf16 colvec pointer, reducing
    #    alignment from 32-bit to 16-bit. The async copy atom requires 32-bit.
    #    Fix: we do NOT monkey-patch here — instead, we pass colvec_scale with
    #    assumed_align=2 (bf16 element size) in the call sites that use varlen.
    #    The actual fix is applied in gemm_dgated.py where we create the cute
    #    tensor from colvec_scale.
    #    Note: this is an upstream QuACK bug (gemm_default_epi.py:121-138).
