"""Low-overhead input validation helpers for GPU kernel wrappers.

All checks are pure Python attribute lookups on tensor metadata —
zero GPU synchronization, zero allocation, typically < 100ns each.

Usage:
    from sonicmoe.quack_utils._validate import check_tensor, check_contiguous_last
    check_tensor(z, "z", dtype=torch.bfloat16, ndim=2, last_stride_1=True)
"""
from __future__ import annotations

import torch


def check_tensor(
    t: torch.Tensor,
    name: str,
    *,
    dtype: torch.dtype | tuple[torch.dtype, ...] | None = None,
    ndim: int | None = None,
    last_stride_1: bool = False,
    stride0_1: bool = False,
) -> None:
    """Validate tensor metadata (CPU-side, zero sync).

    Parameters
    ----------
    t : Tensor
        The tensor to validate.
    name : str
        Human-readable name for error messages.
    dtype : dtype or tuple of dtypes, optional
        Expected dtype(s). If tuple, tensor must match one of them.
    ndim : int, optional
        Expected number of dimensions.
    last_stride_1 : bool
        If True, assert stride(-1) == 1 (contiguous in last dim).
    stride0_1 : bool
        If True, assert stride(0) == 1 (for 1D contiguous tensors).
    """
    if dtype is not None:
        if isinstance(dtype, tuple):
            # Use str comparison for Paddle torch-proxy compatibility
            t_dtype_str = str(t.dtype)
            if t.dtype not in dtype and not any(str(d) == t_dtype_str for d in dtype):
                raise ValueError(
                    f"{name}: expected dtype in {dtype}, got {t.dtype}"
                )
        elif t.dtype != dtype and str(t.dtype) != str(dtype):
            raise ValueError(
                f"{name}: expected dtype {dtype}, got {t.dtype}"
            )
    if ndim is not None and t.ndim != ndim:
        raise ValueError(
            f"{name}: expected ndim={ndim}, got {t.ndim} (shape={t.shape})"
        )
    if last_stride_1 and t.ndim > 0 and t.stride(-1) != 1:
        raise ValueError(
            f"{name}: last dim must be contiguous (stride=1), "
            f"got stride(-1)={t.stride(-1)}"
        )
    if stride0_1 and t.ndim > 0 and t.stride(0) != 1:
        raise ValueError(
            f"{name}: must be contiguous 1D (stride(0)=1), "
            f"got stride(0)={t.stride(0)}"
        )


def check_shape_match(
    t: torch.Tensor,
    name: str,
    expected: tuple[int, ...],
) -> None:
    """Assert tensor has the exact expected shape."""
    if t.shape != expected:
        raise ValueError(
            f"{name}: expected shape {expected}, got {t.shape}"
        )


def check_divisible(value: int, divisor: int, name: str) -> None:
    """Assert an integer is divisible by a given divisor."""
    if value % divisor != 0:
        raise ValueError(
            f"{name}: {value} must be divisible by {divisor}"
        )
