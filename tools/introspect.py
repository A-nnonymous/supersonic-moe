#!/usr/bin/env python3
"""SonicMoE Introspection Engine — Zero-Code-Change Model Analysis.

Performs an instrumented dry-run of the MoE forward+backward pass.
Extracts buffer lifecycle, memory trajectory, kernel timing, and
precision data into a structured ``manifest.json`` consumed by the
visualization suite (``python -m visualization``).

**Non-invasive**: no changes to sonicmoe/ source code.  All
instrumentation uses PyTorch's public hook APIs + monkey-patching
of autograd Function boundaries.

Modes
-----
  trace   — shapes / dtypes / lifecycle / memory (~3 s)
  profile — trace + kernel timing via torch.profiler (~30 s)
  full    — trace + profile + precision audit (~60 s)

Usage
-----
    python tools/introspect.py                        # trace mode
    python tools/introspect.py --mode profile         # + kernel timing
    python tools/introspect.py --mode full            # everything

Output: ``manifest.json`` at repo root.
"""
from __future__ import annotations

import argparse
import collections
import gc
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.json"
MANIFEST_VERSION = 1

# Default Ernie shape
SHAPE = {"T": 8192, "H": 3072, "I": 1536, "E": 8, "K": 8}

# Map _log_stage_memory stage names → visualization phase IDs (0-5)
STAGE_TO_PHASE = {
    "forward:router-metadata": 0,
    "forward:up-proj": 1,
    "forward:fp8-boundary": 1,
    "forward:down-proj-router": 2,
    "backward:down-proj-dgated": 3,
    "backward:down-proj-weight": 3,
    "backward:down-proj-postact-release": 3,
    "backward:up-proj-core": 4,
    "backward:token-reduce": 5,
}

PHASE_NAMES = [
    "Router & Meta",
    "UpProj Fwd",
    "DnProj Fwd",
    "DnProj Bwd",
    "UpBwd (wgrad)",
    "UpBwd (actgrad)",
]

# UpProjection.forward ctx.save_for_backward ordering (8 tensors)
_UP_SAVE_NAMES = [
    "x", "w1", "b1", "expert_frequency_offset",
    "x_gather_idx", "s_scatter_idx", "s_reverse_scatter_idx",
    "num_activated_expert_per_token_offset",
]

# DownProjection.forward ctx.save_for_backward ordering — BF16 (8 tensors)
_DOWN_SAVE_NAMES_BF16 = [
    "z", "w2", "b2", "topk_scores",
    "expert_frequency_offset", "x_gather_idx",
    "s_scatter_idx", "s_reverse_scatter_idx",
]

# DownProjection.forward ctx.save_for_backward ordering — FP8 (9 tensors)
_DOWN_SAVE_NAMES_FP8 = [
    "z_fp8", "z_raw_scales", "w2", "b2", "topk_scores",
    "expert_frequency_offset", "x_gather_idx",
    "s_scatter_idx", "s_reverse_scatter_idx",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TensorRecord:
    """Lifecycle record for a single GPU tensor."""
    name: str
    dtype: str
    shape: list[int]
    size_mib: float
    role: str = "activation"        # activation | weight | grad | scale | index
    create_phase: int = -1
    free_phase: int = -1            # last phase where the tensor is used
    events: list[str] = field(default_factory=list)


@dataclass
class PhaseMemory:
    """Memory snapshot at a phase boundary."""
    phase_id: int
    phase_name: str
    allocated_mib: float = 0.0
    peak_mib: float = 0.0
    reserved_mib: float = 0.0


@dataclass
class KernelRecord:
    """Single kernel timing record."""
    name: str
    category: str
    cuda_time_us: float
    count: int = 1


@dataclass
class ModeManifest:
    """Manifest data for a single mode (bf16 or fp8)."""
    mode: str
    tensors: list[TensorRecord] = field(default_factory=list)
    phase_memory: list[PhaseMemory] = field(default_factory=list)
    kernels: list[KernelRecord] = field(default_factory=list)
    memory_trajectory: dict[str, float] = field(default_factory=dict)
    total_cuda_us: float = 0.0
    wall_clock_ms: float = 0.0
    precision_matrix: list[list[int]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: dtype to precision code (for precision matrix)
# ═══════════════════════════════════════════════════════════════════════════════

_DTYPE_CODE = {
    "torch.bfloat16": 1,
    "torch.float8_e4m3fn": 2,
    "torch.float32": 3,
    "torch.int32": 4,
    "torch.uint8": 5,     # ISA-packed scales
}


def _dtype_to_role(dtype_str: str) -> str:
    """Infer tensor role from dtype string."""
    if "float8" in dtype_str or "uint8" in dtype_str:
        return "scale" if "uint8" in dtype_str else "activation"
    if "int32" in dtype_str:
        return "index"
    if "float32" in dtype_str:
        return "activation"
    return "activation"


def _tensor_size_mib(t) -> float:
    """Tensor size in MiB."""
    if t is None:
        return 0.0
    return t.nelement() * t.element_size() / (1024 ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class PhaseTracker:
    """Tracks the current visualization phase by intercepting _log_stage_memory.

    Non-invasive: monkey-patches the module-level functions in
    ``sonicmoe.functional`` to capture stage transitions.
    """

    def __init__(self):
        self.current_phase: int = -1
        self.stage_log: list[tuple[str, int, dict]] = []
        self._memory_at_phase: dict[int, PhaseMemory] = {}
        self._installed = False
        self._orig_log = None
        self._orig_reset = None
        self._orig_debug_enabled = None

    def install(self):
        """Monkey-patch sonicmoe.functional stage-memory functions."""
        import sonicmoe.functional as F
        self._orig_log = F._log_stage_memory
        self._orig_reset = F._reset_stage_memory_probe
        self._orig_debug_enabled = F._stage_memory_debug_enabled

        tracker = self

        def _patched_debug_enabled() -> bool:
            return True  # always enabled during introspection

        def _patched_reset() -> None:
            import torch
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def _patched_log(stage: str) -> None:
            import torch
            torch.cuda.synchronize()
            mib = 1024 ** 2
            phase_id = STAGE_TO_PHASE.get(stage, tracker.current_phase)
            tracker.current_phase = phase_id

            mem = {
                "allocated_mib": torch.cuda.memory_allocated() / mib,
                "peak_mib": torch.cuda.max_memory_allocated() / mib,
                "reserved_mib": torch.cuda.memory_reserved() / mib,
            }
            tracker.stage_log.append((stage, phase_id, mem))

            if phase_id not in tracker._memory_at_phase or \
               mem["peak_mib"] > tracker._memory_at_phase[phase_id].peak_mib:
                tracker._memory_at_phase[phase_id] = PhaseMemory(
                    phase_id=phase_id,
                    phase_name=PHASE_NAMES[phase_id] if phase_id < len(PHASE_NAMES) else f"phase_{phase_id}",
                    allocated_mib=round(mem["allocated_mib"], 2),
                    peak_mib=round(mem["peak_mib"], 2),
                    reserved_mib=round(mem["reserved_mib"], 2),
                )

        F._stage_memory_debug_enabled = _patched_debug_enabled
        F._log_stage_memory = _patched_log
        F._reset_stage_memory_probe = _patched_reset
        self._installed = True

    def uninstall(self):
        """Restore original functions."""
        if not self._installed:
            return
        import sonicmoe.functional as F
        F._log_stage_memory = self._orig_log
        F._reset_stage_memory_probe = self._orig_reset
        F._stage_memory_debug_enabled = self._orig_debug_enabled
        self._installed = False

    def get_phase_memory(self) -> list[PhaseMemory]:
        """Return memory snapshots sorted by phase ID."""
        return [self._memory_at_phase[k] for k in sorted(self._memory_at_phase)]

    def reset(self):
        """Clear state for a new trace run."""
        self.current_phase = -1
        self.stage_log.clear()
        self._memory_at_phase.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Tensor Spy
# ═══════════════════════════════════════════════════════════════════════════════

class TensorSpy:
    """Tracks tensor lifecycle using saved_tensors_hooks + forward/backward wrapping.

    Records: name, dtype, shape, size, create_phase, last_used_phase, role.
    """

    def __init__(self, phase_tracker: PhaseTracker):
        self.phase_tracker = phase_tracker
        self._ptr_to_name: dict[int, str] = {}      # data_ptr → name
        self._records: dict[str, TensorRecord] = {}  # name → TensorRecord
        self._pack_count = 0
        self._unpack_count = 0
        self._prequant_events: list[tuple[str, str, int]] = []  # (key, action, phase)

        # Monkey-patching state
        self._orig_up_fwd = None
        self._orig_up_bwd = None
        self._orig_down_fwd = None
        self._orig_down_bwd = None
        self._prequant_proxy = None
        self._installed = False

    def _record_tensor(self, name: str, tensor, phase: int,
                       role: str = "activation", event: str | None = None):
        """Register or update a tensor record."""
        if tensor is None:
            return
        import torch
        if not isinstance(tensor, torch.Tensor):
            return

        dtype_str = str(tensor.dtype)
        size_mib = round(_tensor_size_mib(tensor), 4)
        shape = list(tensor.shape)

        ptr = tensor.data_ptr()

        # --- Deduplication by data_ptr ---
        # The CUDA caching allocator reuses freed memory, so different tensors
        # at different phases may share a data_ptr.  Distinguish same-tensor
        # (e.g. topk_indices passed as selected_experts) from ptr-reuse by
        # checking shape+dtype match.  If shape/dtype differ, it's a new alloc.
        old_name = self._ptr_to_name.get(ptr)
        if old_name and old_name != name:
            old_rec = self._records.get(old_name)
            if old_rec is not None:
                same_shape_dtype = (old_rec.shape == shape
                                    and old_rec.dtype == dtype_str)
                if not same_shape_dtype:
                    # Different shape/dtype → definitely a data_ptr reuse
                    del self._ptr_to_name[ptr]
                    old_name = None
            if old_name and old_name.startswith(("saved_", "restored_")):
                # Auto-name → meaningful name: absorb old record
                old_rec_pop = self._records.pop(old_name, None)
                if old_rec_pop:
                    if name in self._records:
                        rec = self._records[name]
                        rec.create_phase = min(rec.create_phase,
                                               old_rec_pop.create_phase)
                        rec.free_phase = max(rec.free_phase,
                                             old_rec_pop.free_phase)
                        rec.events.extend(old_rec_pop.events)
                    else:
                        old_rec_pop.name = name
                        if role != "activation":
                            old_rec_pop.role = role
                        self._records[name] = old_rec_pop
            elif old_name and not name.startswith(("saved_", "restored_")):
                # Both meaningful — same physical tensor under different var
                # names (e.g. topk_indices == selected_experts).  Keep the
                # first-registered name and just extend its lifecycle.
                name = old_name

        if name in self._records:
            rec = self._records[name]
            if phase < rec.create_phase or rec.create_phase < 0:
                rec.create_phase = phase
            if phase > rec.free_phase:
                rec.free_phase = phase
            if event:
                rec.events.append(event)
        else:
            self._records[name] = TensorRecord(
                name=name,
                dtype=dtype_str,
                shape=shape,
                size_mib=size_mib,
                role=role,
                create_phase=phase,
                free_phase=phase,
                events=[event] if event else [],
            )
        self._ptr_to_name[ptr] = name

    def _infer_name_from_shape(self, tensor, phase: int) -> str | None:
        """Heuristic naming for tensors not caught by wrapper registration."""
        import torch
        shape = list(tensor.shape)
        dtype = tensor.dtype

        # Phase 0: Router tensors
        if phase == 0:
            if dtype == torch.float32 and len(shape) == 2 and shape[1] <= 32:
                return "topk_scores"   # [T, K] FP32
            if dtype == torch.int32 and len(shape) == 2 and shape[1] <= 32:
                return "topk_indices"  # [T, K] INT32

        # Phase 2: Aux loss intermediates (tiny FP32 tensors from _compute_switch_loss)
        if phase == 2 and dtype == torch.float32:
            if len(shape) == 1 and shape[0] == 1:
                return "aux_loss_norm"
            if len(shape) == 1 and shape[0] <= 32:
                n = sum(1 for k in self._records if k and k.startswith("aux_"))
                return f"aux_loss_{n}"
            if len(shape) == 2 and shape[1] <= 32:
                return "softmax_probs"  # [T, E] FP32 from F.softmax

        return None

    def _make_pack_hook(self):
        spy = self
        def pack_hook(tensor):
            ptr = tensor.data_ptr()
            phase = spy.phase_tracker.current_phase
            name = spy._ptr_to_name.get(ptr)
            if name is None:
                name = spy._infer_name_from_shape(tensor, phase)
            if name is None:
                name = f"saved_{spy._pack_count}"
            spy._record_tensor(name, tensor, phase, event=f"saved@phase{phase}")
            spy._pack_count += 1
            return tensor
        return pack_hook

    def _make_unpack_hook(self):
        spy = self
        def unpack_hook(tensor):
            ptr = tensor.data_ptr()
            phase = spy.phase_tracker.current_phase
            name = spy._ptr_to_name.get(ptr, f"restored_{spy._unpack_count}")
            spy._record_tensor(name, tensor, phase, event=f"restored@phase{phase}")
            spy._unpack_count += 1
            return tensor
        return unpack_hook

    def install(self, model):
        """Install all hooks and monkey-patches."""
        import torch
        from sonicmoe.functional import _UpProjection, _DownProjection

        # 1. Register all model parameters
        for pname, param in model.named_parameters():
            short = self._param_short_name(pname)
            self._record_tensor(short, param.data, phase=0, role="weight")

        # 2. Wrap _UpProjection.forward to capture arg names
        spy = self
        self._orig_up_fwd = _UpProjection.forward

        @staticmethod
        def _traced_up_fwd(ctx, x, w1, b1, expert_frequency_offset,
                           total_expert_freq, K, stream_id, x_gather_idx,
                           s_scatter_idx, s_reverse_scatter_idx,
                           num_activated_expert_per_token_offset,
                           is_varlen_K, activation_type,
                           is_inference_mode_enabled,
                           use_low_precision_postact_buffer):
            phase = 1
            spy.phase_tracker.current_phase = phase
            # Record forward args
            spy._record_tensor("x", x, phase)
            spy._record_tensor("w1", w1, phase, role="weight")
            spy._record_tensor("x_gather_idx", x_gather_idx, phase, role="index")
            spy._record_tensor("s_scatter_idx", s_scatter_idx, phase, role="index")
            spy._record_tensor("s_reverse_scatter_idx", s_reverse_scatter_idx, phase, role="index")
            spy._record_tensor("expert_freq_offset", expert_frequency_offset, phase, role="index")
            spy._record_tensor("num_activated_expert_offset",
                               num_activated_expert_per_token_offset, phase, role="index")

            result = spy._orig_up_fwd(
                ctx, x, w1, b1, expert_frequency_offset,
                total_expert_freq, K, stream_id, x_gather_idx,
                s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                is_varlen_K, activation_type,
                is_inference_mode_enabled,
                use_low_precision_postact_buffer,
            )

            y1, z = result
            spy._record_tensor("y1", y1, phase, event="created@UpProj")
            spy._record_tensor("z", z, phase, event="created@UpProj")
            return result

        _UpProjection.forward = _traced_up_fwd

        # 3. Wrap _UpProjection.backward
        self._orig_up_bwd = _UpProjection.backward

        @staticmethod
        def _traced_up_bwd(ctx, grad_y1, dz):
            spy.phase_tracker.current_phase = 4
            if dz is not None:
                spy._record_tensor("dz", dz, 4, role="grad")

            result = spy._orig_up_bwd(ctx, grad_y1, dz)

            # result = (dx_reduced, dw1, db1, None×12)
            if result[0] is not None:
                spy._record_tensor("dx", result[0], 5, role="grad")
            if result[1] is not None:
                spy._record_tensor("dw1", result[1], 4, role="grad")
            spy.phase_tracker.current_phase = 5
            return result

        _UpProjection.backward = _traced_up_bwd

        # 4. Wrap _DownProjection.forward
        self._orig_down_fwd = _DownProjection.forward

        @staticmethod
        def _traced_down_fwd(ctx, y1, z, w2, b2, topk_scores,
                             selected_experts, expert_frequency_offset,
                             T, K, stream_id, x_gather_idx,
                             s_scatter_idx, s_reverse_scatter_idx,
                             num_activated_expert_per_token_offset,
                             is_varlen_K, activation_type, fp8_protocol):
            phase = 2
            spy.phase_tracker.current_phase = phase
            spy._record_tensor("y1", y1, phase)
            spy._record_tensor("z", z, phase)
            spy._record_tensor("w2", w2, phase, role="weight")
            spy._record_tensor("topk_scores", topk_scores, phase)
            spy._record_tensor("selected_experts", selected_experts, phase, role="index")
            spy._record_tensor("expert_freq_offset", expert_frequency_offset, phase, role="index")
            spy._record_tensor("x_gather_idx", x_gather_idx, phase, role="index")
            spy._record_tensor("s_scatter_idx", s_scatter_idx, phase, role="index")
            spy._record_tensor("s_reverse_scatter_idx", s_reverse_scatter_idx, phase, role="index")

            result = spy._orig_down_fwd(
                ctx, y1, z, w2, b2, topk_scores,
                selected_experts, expert_frequency_offset,
                T, K, stream_id, x_gather_idx,
                s_scatter_idx, s_reverse_scatter_idx,
                num_activated_expert_per_token_offset,
                is_varlen_K, activation_type, fp8_protocol,
            )

            spy._record_tensor("output", result, phase, event="DnProj_out")
            return result

        _DownProjection.forward = _traced_down_fwd

        # 5. Wrap _DownProjection.backward
        self._orig_down_bwd = _DownProjection.backward

        @staticmethod
        def _traced_down_bwd(ctx, dout):
            spy.phase_tracker.current_phase = 3
            spy._record_tensor("dout", dout, 3, role="grad")

            result = spy._orig_down_bwd(ctx, dout)

            # result = (dy1, dz, dw2, db2, ..., Nones)
            if result[1] is not None:
                spy._record_tensor("dz", result[1], 3, role="grad",
                                   event="created@DnProj_bwd")
            if result[2] is not None:
                spy._record_tensor("dw2", result[2], 3, role="grad")
            return result

        _DownProjection.backward = _traced_down_bwd

        # 6. Monitor _PREQUANTIZED_SCALES via monkey-patched dict
        from sonicmoe.functional import _PREQUANTIZED_SCALES
        self._install_prequant_monitor(_PREQUANTIZED_SCALES)

        self._installed = True

    def _install_prequant_monitor(self, original_dict):
        """Wrap _PREQUANTIZED_SCALES dict to monitor FP8 tensor caching."""
        spy = self
        import sonicmoe.functional as F

        class _MonitoredDict(dict):
            def __setitem__(self, key, value):
                phase = spy.phase_tracker.current_phase
                spy._prequant_events.append((key, "store", phase))
                # Record FP8 tensors from the cache
                import torch
                if isinstance(value, tuple):
                    for i, v in enumerate(value):
                        if isinstance(v, torch.Tensor):
                            suffix = f"_{i}" if len(value) > 1 else ""
                            tname = f"prequant_{key}{suffix}"
                            dtype_str = str(v.dtype)
                            if "float8" in dtype_str:
                                tname = f"{key}" if "fp8" in key else f"{key}_fp8"
                            elif "uint8" in dtype_str:
                                tname = f"{key}_scales"
                            spy._record_tensor(tname, v, phase,
                                               role="scale" if "scale" in tname else "activation",
                                               event=f"prequant_store@phase{phase}")
                super().__setitem__(key, value)

            def pop(self, key, *args):
                phase = spy.phase_tracker.current_phase
                spy._prequant_events.append((key, "consume", phase))
                return super().pop(key, *args)

        monitored = _MonitoredDict(original_dict)
        monitored.update(original_dict)
        F._PREQUANTIZED_SCALES = monitored
        self._prequant_proxy = monitored

    def uninstall(self):
        """Restore all original methods."""
        if not self._installed:
            return
        from sonicmoe.functional import _UpProjection, _DownProjection
        import sonicmoe.functional as F

        if self._orig_up_fwd is not None:
            _UpProjection.forward = self._orig_up_fwd
        if self._orig_up_bwd is not None:
            _UpProjection.backward = self._orig_up_bwd
        if self._orig_down_fwd is not None:
            _DownProjection.forward = self._orig_down_fwd
        if self._orig_down_bwd is not None:
            _DownProjection.backward = self._orig_down_bwd

        # Restore original PREQUANTIZED_SCALES dict
        if self._prequant_proxy is not None:
            original = dict(self._prequant_proxy)
            F._PREQUANTIZED_SCALES = original
        self._installed = False

    def get_records(self) -> list[TensorRecord]:
        """Return all tracked tensor records sorted by create_phase."""
        recs = sorted(self._records.values(), key=lambda r: (r.create_phase, r.name))
        return recs

    def reset(self):
        """Clear state for a new run."""
        self._ptr_to_name.clear()
        self._records.clear()
        self._pack_count = 0
        self._unpack_count = 0
        self._prequant_events.clear()

    @staticmethod
    def _param_short_name(full_name: str) -> str:
        """Map model parameter name to visualization name."""
        mapping = {
            "router.weight": "router_w",
            "c_fc.weight": "w1",
            "c_fc.bias": "b1",
            "c_proj.weight": "w2",
            "c_proj.bias": "b2",
        }
        return mapping.get(full_name, full_name)


# ═══════════════════════════════════════════════════════════════════════════════
# Precision Matrix Builder
# ═══════════════════════════════════════════════════════════════════════════════

def _build_precision_matrix(tensors: list[TensorRecord], n_phases: int = 6) -> list[list[int]]:
    """Build 13×6 precision matrix from tensor records.

    Encoding: 0=absent, 1=BF16, 2=FP8, 3=FP32, 4=INT32, 5=SCALE/UINT8.
    """
    matrix = []
    for t in tensors:
        row = []
        code = _DTYPE_CODE.get(t.dtype, 0)
        for p in range(n_phases):
            if t.create_phase <= p <= t.free_phase:
                row.append(code)
            else:
                row.append(0)
        matrix.append(row)
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Trajectory
# ═══════════════════════════════════════════════════════════════════════════════

def _capture_memory_trajectory(model, x, use_fp8: bool) -> dict[str, float]:
    """Run forward + backward and capture memory at 4 key checkpoints."""
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    mib = 1024 ** 2

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    pre_fwd = torch.cuda.memory_allocated() / mib

    x_run = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                out, loss_val = model(x_run, use_fp8=True)
        else:
            out, loss_val = model(x_run)
    torch.cuda.synchronize()
    peak_fwd = torch.cuda.max_memory_allocated() / mib

    torch.cuda.reset_peak_memory_stats()
    pre_bwd = torch.cuda.memory_allocated() / mib
    loss = out.sum() + loss_val
    loss.backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated() / mib

    del out, loss, loss_val, x_run
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cleanup = torch.cuda.memory_allocated() / mib

    return {
        "pre_fwd": round(pre_fwd, 2),
        "peak_fwd": round(peak_fwd, 2),
        "pre_bwd": round(pre_bwd, 2),
        "peak_bwd": round(peak_bwd, 2),
        "cleanup": round(cleanup, 2),
        "fwd_peak_above_pre": round(peak_fwd - pre_fwd, 2),
        "bwd_peak_above_pre": round(peak_bwd - pre_bwd, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Trace Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_trace(mode: str, model, x, device,
              phase_tracker: PhaseTracker,
              tensor_spy: TensorSpy) -> ModeManifest:
    """Run an instrumented forward+backward pass, capture tensor lifecycle + memory.

    Parameters
    ----------
    mode : str
        "bf16" or "fp8"
    model : MoE
        The model to trace.
    x : torch.Tensor
        Input tensor [T, H].
    device : torch.device
        CUDA device.
    phase_tracker : PhaseTracker
        Installed phase tracker.
    tensor_spy : TensorSpy
        Installed tensor spy.

    Returns
    -------
    ModeManifest
        Trace data for this mode.
    """
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    use_fp8 = (mode == "fp8")

    # Reset trackers
    phase_tracker.reset()
    tensor_spy.reset()

    # Re-register parameters (they may have been cleared)
    for pname, param in model.named_parameters():
        short = tensor_spy._param_short_name(pname)
        tensor_spy._record_tensor(short, param.data, phase=0, role="weight")

    # Clean CUDA state
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # Run instrumented forward + backward with saved_tensors_hooks
    x_run = x.detach().clone().requires_grad_(True)

    # Pre-register input so F.linear's autograd save finds the correct name
    tensor_spy._record_tensor("x", x_run, phase=0)

    with torch.autograd.graph.saved_tensors_hooks(
        tensor_spy._make_pack_hook(),
        tensor_spy._make_unpack_hook(),
    ):
        phase_tracker.current_phase = 0
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    out, loss_val = model(x_run, use_fp8=True)
            else:
                out, loss_val = model(x_run)
        torch.cuda.synchronize()

        # Backward
        loss = out.sum() + loss_val
        loss.backward()
        torch.cuda.synchronize()

    # Collect results
    manifest = ModeManifest(mode=mode)
    manifest.tensors = tensor_spy.get_records()
    manifest.phase_memory = phase_tracker.get_phase_memory()
    manifest.precision_matrix = _build_precision_matrix(manifest.tensors)

    # Capture memory trajectory (clean run without hooks)
    del out, loss, loss_val, x_run
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    manifest.memory_trajectory = _capture_memory_trajectory(model, x, use_fp8)

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Kernel Profiler (subprocess-isolated)
# ═══════════════════════════════════════════════════════════════════════════════

_KERNEL_PROFILE_SCRIPT = r'''
import gc, json, os, sys, time, torch
from collections import defaultdict

MODE = os.environ["_PROFILER_MODE"]
T, H, I, E, K = {T}, {H}, {I}, {E}, {K}
device = torch.device("cuda:0")
torch.cuda.set_device(device)

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
torch.manual_seed(42)
model = MoE(E, K, H, I, ActivationType.SWIGLU, False, 0.02).to(device).to(torch.bfloat16)
x = 0.02 * torch.randn(T, H, dtype=torch.bfloat16, device=device, requires_grad=True)

use_fp8 = (MODE == "fp8")

# Warmup
for _ in range(5):
    xw = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                ow, lw = model(xw, use_fp8=True)
        else:
            ow, lw = model(xw)
    (ow.sum() + lw).backward()
    model.zero_grad(set_to_none=True)
    del ow, lw, xw
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# Profile
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(3):
        xp = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    op, lp = model(xp, use_fp8=True)
            else:
                op, lp = model(xp)
        (op.sum() + lp).backward()
        model.zero_grad(set_to_none=True)
        del op, lp, xp
    torch.cuda.synchronize()

# Wall-clock timing
times = []
for _ in range(20):
    xt = x.detach().clone().requires_grad_(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with enable_quack_gemm(True):
        if use_fp8:
            with enable_fp8(True):
                ot, lt = model(xt, use_fp8=True)
        else:
            ot, lt = model(xt)
    (ot.sum() + lt).backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
    model.zero_grad(set_to_none=True)
    del ot, lt, xt
gc.collect()

# Aggregate kernels
kernel_agg = defaultdict(lambda: {{"total_us": 0.0, "count": 0}})
for evt in prof.key_averages():
    if hasattr(evt, "self_device_time_total") and evt.self_device_time_total > 0:
        kernel_agg[evt.key]["total_us"] += evt.self_device_time_total / 3  # 3 iters
        kernel_agg[evt.key]["count"] += evt.count // 3

# Sort by time
sorted_kernels = sorted(kernel_agg.items(), key=lambda kv: -kv[1]["total_us"])

total_cuda = sum(v["total_us"] for v in kernel_agg.values())

result = {{
    "mode": MODE,
    "total_cuda_us": round(total_cuda, 1),
    "wall_clock_ms": round(sum(times[5:]) / len(times[5:]), 3),
    "kernels": [
        {{"name": k, "cuda_time_us": round(v["total_us"], 2), "count": v["count"]}}
        for k, v in sorted_kernels if v["total_us"] > 1.0
    ],
}}
print("__KERNEL_JSON__" + json.dumps(result))
'''


def _categorize_kernel(name: str) -> str:
    """Map torch.profiler kernel name to human-readable category."""
    if "GemmDefault" in name and "Sm100" in name:
        return "Wgrad GEMM"
    if "GemmGated" in name and "ZeroMat" not in name:
        return "GemmGated (fwd)"
    if "GemmDGated" in name and "ZeroMat" not in name:
        return "GemmDGated (bwd)"
    if "GemmGated" in name and "ZeroMat" in name:
        return "GemmGated ZeroMat (fwd)"
    if "GemmDGated" in name and "ZeroMat" in name:
        return "GemmDGated ZeroMat (bwd)"
    if "blockscaled_quant" in name.lower() or "BlockscaledQuant" in name:
        return "Blockscaled Quant"
    if "flat_quant" in name.lower() or "FlatQuant" in name:
        return "Flat Quant"
    if "gather_isa" in name.lower() or "ISAGather" in name:
        return "ISA Scale Gather"
    if "swiglu" in name.lower() or "SwiGLU" in name:
        return "SwiGLU"
    if "scatter" in name.lower() and "token" in name.lower():
        return "Token Scatter"
    if "topk" in name.lower():
        return "TopK Router"
    if "softmax" in name.lower():
        return "Softmax"
    return "Other"


def run_kernel_profile() -> dict[str, Any]:
    """Run kernel profiling in subprocess for both BF16 and FP8.

    Returns dict with "bf16" and "fp8" kernel data.
    """
    env_path = "/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/envs/xfer"
    python_bin = os.path.join(env_path, "bin", "python")

    result = {}
    for mode in ("bf16", "fp8"):
        script = _KERNEL_PROFILE_SCRIPT.format(**SHAPE)
        env = os.environ.copy()
        env["_PROFILER_MODE"] = mode
        env["USE_QUACK_GEMM"] = "1"
        if mode == "fp8":
            env["SONIC_MOE_FP8_MODE"] = "perf"

        print(f"  Kernel profiling [{mode}] in subprocess ...", flush=True)
        try:
            proc = subprocess.run(
                [python_bin, "-c", script],
                capture_output=True, text=True, timeout=120, env=env,
                cwd=str(ROOT),
            )
            # Extract JSON from output
            for line in proc.stdout.split("\n"):
                if line.startswith("__KERNEL_JSON__"):
                    data = json.loads(line[len("__KERNEL_JSON__"):])
                    # Add categories
                    for k in data.get("kernels", []):
                        k["category"] = _categorize_kernel(k["name"])
                    result[mode] = data
                    break
            else:
                print(f"  WARNING: No kernel JSON found in {mode} output", flush=True)
                if proc.stderr:
                    print(f"  stderr (last 500 chars): {proc.stderr[-500:]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: Kernel profiling [{mode}] timed out", flush=True)
        except Exception as e:
            print(f"  WARNING: Kernel profiling [{mode}] failed: {e}", flush=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Precision Auditor
# ═══════════════════════════════════════════════════════════════════════════════

def run_precision_audit(model, x) -> dict[str, Any]:
    """Compare BF16 vs FP8 outputs: RRMSE and cosine similarity."""
    import torch
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm

    def _run(use_fp8: bool):
        x_run = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            if use_fp8:
                with enable_fp8(True):
                    out, loss_val = model(x_run, use_fp8=True)
            else:
                out, loss_val = model(x_run)
        loss = out.sum() + loss_val
        loss.backward()
        result = {
            "output": out.detach().float(),
            "dx": x_run.grad.detach().float() if x_run.grad is not None else None,
        }
        # Collect weight gradients
        for pname, p in model.named_parameters():
            if p.grad is not None:
                result[f"grad_{pname}"] = p.grad.detach().float()
        model.zero_grad(set_to_none=True)
        del out, loss, loss_val
        gc.collect()
        return result

    print("  Running BF16 baseline ...", flush=True)
    bf16_out = _run(False)
    print("  Running FP8 frontier ...", flush=True)
    fp8_out = _run(True)

    def _rrmse(a, b):
        if a is None or b is None:
            return None
        diff = (a - b).float()
        return float((diff.norm() / b.float().norm() * 100).item())

    def _cosine(a, b):
        if a is None or b is None:
            return None
        a_flat, b_flat = a.flatten().float(), b.flatten().float()
        return float(torch.nn.functional.cosine_similarity(
            a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())

    audit = {"rrmse_pct": {}, "cosine_sim": {}}

    for key in ["output", "dx"]:
        audit["rrmse_pct"][key] = round(_rrmse(fp8_out.get(key), bf16_out.get(key)), 4) \
            if bf16_out.get(key) is not None else None
        audit["cosine_sim"][key] = round(_cosine(fp8_out.get(key), bf16_out.get(key)), 6) \
            if bf16_out.get(key) is not None else None

    # Weight gradients
    for pname in ["c_fc.weight", "c_proj.weight"]:
        gkey = f"grad_{pname}"
        short = "dw1" if "c_fc" in pname else "dw2"
        audit["rrmse_pct"][short] = round(_rrmse(fp8_out.get(gkey), bf16_out.get(gkey)), 4) \
            if bf16_out.get(gkey) is not None else None

    del bf16_out, fp8_out
    gc.collect()
    return audit


# ═══════════════════════════════════════════════════════════════════════════════
# Manifest Writer
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_manifest(
    bf16_manifest: ModeManifest | None,
    fp8_manifest: ModeManifest | None,
    kernel_data: dict | None,
    precision_audit: dict | None,
    metadata: dict | None = None,
) -> dict:
    """Assemble all data into the final manifest dict."""
    manifest = {
        "version": MANIFEST_VERSION,
        "metadata": metadata or {},
        "modes": {},
    }

    for m in [bf16_manifest, fp8_manifest]:
        if m is None:
            continue
        mode_dict = {
            "tensors": [asdict(t) for t in m.tensors],
            "phase_memory": [asdict(pm) for pm in m.phase_memory],
            "memory_trajectory": m.memory_trajectory,
            "precision_matrix": m.precision_matrix,
        }

        # Merge kernel data if available
        if kernel_data and m.mode in kernel_data:
            kd = kernel_data[m.mode]
            mode_dict["kernels"] = kd.get("kernels", [])
            mode_dict["total_cuda_us"] = kd.get("total_cuda_us", 0)
            mode_dict["wall_clock_ms"] = kd.get("wall_clock_ms", 0)

        manifest["modes"][m.mode] = mode_dict

    # GPU projection speedup
    if kernel_data and "bf16" in kernel_data and "fp8" in kernel_data:
        bf16_cuda = kernel_data["bf16"].get("total_cuda_us", 0)
        fp8_cuda = kernel_data["fp8"].get("total_cuda_us", 0)
        if fp8_cuda > 0:
            manifest["gpu_projection_speedup"] = round(bf16_cuda / fp8_cuda, 4)

    if precision_audit:
        manifest["precision_audit"] = precision_audit

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run(mode: str = "trace") -> dict:
    """Main entry point: run introspection and write manifest.json.

    Parameters
    ----------
    mode : str
        "trace" | "profile" | "full"
    """
    import torch

    print("=" * 60)
    print(f"SonicMoE Introspection Engine  [mode={mode}]")
    print("=" * 60)

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Metadata
    gpu_name = torch.cuda.get_device_name(device)
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shape": SHAPE,
        "device": gpu_name,
        "torch_version": torch.__version__,
        "mode": mode,
    }
    try:
        import quack
        metadata["quack_version"] = getattr(quack, "__version__", "unknown")
    except ImportError:
        metadata["quack_version"] = "not installed"

    print(f"  Device: {gpu_name}")
    print(f"  Shape: T={SHAPE['T']}, H={SHAPE['H']}, I={SHAPE['I']}, "
          f"E={SHAPE['E']}, K={SHAPE['K']}")

    # Create model + input
    from sonicmoe import MoE
    from sonicmoe.enums import ActivationType

    torch.manual_seed(42)
    model = MoE(
        SHAPE["E"], SHAPE["K"], SHAPE["H"], SHAPE["I"],
        ActivationType.SWIGLU, False, 0.02,
    ).to(device).to(torch.bfloat16)

    x = 0.02 * torch.randn(
        SHAPE["T"], SHAPE["H"],
        dtype=torch.bfloat16, device=device, requires_grad=True,
    )

    # Warmup
    print("\n[1/5] Warmup ...", flush=True)
    from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
    for _ in range(3):
        xw = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True):
            ow, lw = model(xw)
        (ow.sum() + lw).backward()
        model.zero_grad(set_to_none=True)
        del ow, lw, xw
    gc.collect()
    torch.cuda.empty_cache()

    # Install hooks
    phase_tracker = PhaseTracker()
    tensor_spy = TensorSpy(phase_tracker)
    phase_tracker.install()
    tensor_spy.install(model)

    # ── Trace BF16 ──
    print("\n[2/5] Tracing BF16 ...", flush=True)
    bf16_manifest = run_trace("bf16", model, x, device, phase_tracker, tensor_spy)
    print(f"       → {len(bf16_manifest.tensors)} tensors tracked, "
          f"{len(bf16_manifest.phase_memory)} phase snapshots")

    # FP8 warmup (need to prime caches)
    for _ in range(3):
        xw = x.detach().clone().requires_grad_(True)
        with enable_quack_gemm(True), enable_fp8(True):
            ow, lw = model(xw, use_fp8=True)
        (ow.sum() + lw).backward()
        model.zero_grad(set_to_none=True)
        del ow, lw, xw
    gc.collect()
    torch.cuda.empty_cache()

    # ── Trace FP8 ──
    print("\n[3/5] Tracing FP8 ...", flush=True)
    fp8_manifest = run_trace("fp8", model, x, device, phase_tracker, tensor_spy)
    print(f"       → {len(fp8_manifest.tensors)} tensors tracked, "
          f"{len(fp8_manifest.phase_memory)} phase snapshots")

    # Uninstall hooks
    tensor_spy.uninstall()
    phase_tracker.uninstall()

    # ── Kernel profiling (if requested) ──
    kernel_data = None
    if mode in ("profile", "full"):
        print("\n[4/5] Kernel profiling (subprocess) ...", flush=True)
        kernel_data = run_kernel_profile()
        for m in ("bf16", "fp8"):
            if m in (kernel_data or {}):
                kd = kernel_data[m]
                print(f"       [{m}] {kd['total_cuda_us']:.1f} µs CUDA, "
                      f"{kd['wall_clock_ms']:.2f} ms wall, "
                      f"{len(kd['kernels'])} kernels")
    else:
        print("\n[4/5] Kernel profiling SKIPPED (use --mode profile)", flush=True)
        # Try loading from existing kernel_breakdown.json
        kern_path = ROOT / "kernel_breakdown.json"
        if kern_path.exists():
            print(f"       → Loading cached data from {kern_path.name}")
            cached = json.loads(kern_path.read_text())
            kernel_data = {}
            for m in ("bf16", "fp8"):
                if m in cached:
                    kernel_data[m] = cached[m]

    # ── Precision audit (if full mode) ──
    precision_audit = None
    if mode == "full":
        print("\n[5/5] Precision audit ...", flush=True)
        precision_audit = run_precision_audit(model, x)
        rrmse = precision_audit.get("rrmse_pct", {})
        print(f"       RRMSE: output={rrmse.get('output', '?')}%, "
              f"dx={rrmse.get('dx', '?')}%, "
              f"dw1={rrmse.get('dw1', '?')}%, "
              f"dw2={rrmse.get('dw2', '?')}%")
    else:
        print("\n[5/5] Precision audit SKIPPED (use --mode full)", flush=True)

    # ── Assemble and write manifest ──
    print("\n[WRITE] Assembling manifest ...", flush=True)
    manifest = _serialize_manifest(
        bf16_manifest, fp8_manifest, kernel_data, precision_audit, metadata
    )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))
    size_kb = MANIFEST_PATH.stat().st_size / 1024
    print(f"  → {MANIFEST_PATH} ({size_kb:.1f} KB)")
    print("=" * 60)
    print("Done. Visualization can now consume manifest.json:")
    print("  python -m visualization")
    print("=" * 60)

    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SonicMoE Introspection Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Modes:
              trace   — shapes / dtypes / lifecycle / memory (~3 s)
              profile — trace + kernel timing via torch.profiler (~30 s)
              full    — trace + profile + precision audit (~60 s)

            Example:
              python tools/introspect.py --mode full
        """),
    )
    parser.add_argument(
        "--mode", choices=["trace", "profile", "full"], default="trace",
        help="Introspection depth (default: trace)",
    )
    parser.add_argument(
        "--shape", type=str, default=None,
        help="Override shape as T,H,I,E,K (e.g. '8192,3072,1536,8,8')",
    )
    args = parser.parse_args()

    if args.shape:
        parts = [int(x) for x in args.shape.split(",")]
        assert len(parts) == 5, f"Expected T,H,I,E,K but got {len(parts)} values"
        SHAPE["T"], SHAPE["H"], SHAPE["I"], SHAPE["E"], SHAPE["K"] = parts

    run(mode=args.mode)


if __name__ == "__main__":
    main()
