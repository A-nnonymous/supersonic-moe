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
