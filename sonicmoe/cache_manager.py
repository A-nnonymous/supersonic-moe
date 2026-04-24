"""Unified JIT cache management: Triton, CuTe, in-memory dicts.

Sets up all JIT cache paths to a single user-specified root (NFS/GPFS)
so caches survive container restarts.

Usage:
    SONIC_MOE_CACHE_DIR=/path/to/nfs/cache  # environment variable, or
    from sonicmoe.cache_manager import setup_cache
    setup_cache("/path/to/nfs/cache")        # Python API

Logging:
    Set SONIC_MOE_JIT_VERBOSE=1 to see compile timing / cache hits.
"""

import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path

_log = logging.getLogger("sonicmoe.jit")

DEFAULT_CACHE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".jit_cache",
)

_VERBOSE = os.environ.get("SONIC_MOE_JIT_VERBOSE", "0") == "1"

# Global counter for progress estimation
_compile_count = 0
_compile_start_time: float | None = None
_EXPECTED_KERNEL_COUNT = 45  # ~15 CuTe + ~30 Triton


def _is_verbose() -> bool:
    return _VERBOSE or os.environ.get("SONIC_MOE_JIT_VERBOSE", "0") == "1"


def get_cache_root() -> Path:
    return Path(os.getenv("SONIC_MOE_CACHE_DIR", DEFAULT_CACHE_ROOT))


def setup_cache(cache_root: str | None = None) -> Path:
    """Initialize all JIT caches under a unified root path.

    Subdirectory structure::

        {cache_root}/
          triton/           <- TRITON_CACHE_DIR
          quack/            <- QUACK_CACHE_DIR
          sonicmoe/         <- reserved for future .o persistence
          version.json      <- git hash + timestamp for manual invalidation

    Parameters
    ----------
    cache_root : str, optional
        Override cache root.  Falls back to SONIC_MOE_CACHE_DIR env var,
        then to ``{project_root}/.jit_cache``.

    Returns
    -------
    Path
        The resolved cache root.
    """
    root = Path(cache_root) if cache_root else get_cache_root()
    for sub in ("triton", "quack", "sonicmoe"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    # Triton env vars
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    os.environ["TRITON_HOME"] = str(root)
    os.environ["TRITON_CACHE_AUTOTUNING"] = "1"

    # CuTe/quack env vars
    os.environ["QUACK_CACHE_DIR"] = str(root / "quack")

    # Version info
    _write_version_info(root / "version.json")

    if _is_verbose():
        _log.info(f"[JIT] Cache root: {root}")

    return root


def _write_version_info(path: Path):
    """Record cache version info for manual invalidation."""
    import subprocess

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_hash = "unknown"
    info = {
        "git_hash": git_hash,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        path.write_text(json.dumps(info, indent=2))
    except OSError:
        pass


class InstrumentedCompileCache:
    """Drop-in replacement for ``dict`` compile caches with timing + logging.

    L1: in-memory dict (same as original behavior).
    Adds compile timing, cache hit/miss logging, and progress estimation.

    Usage (replaces ``_COMPILE_CACHE: dict = {}``)::

        _COMPILE_CACHE = InstrumentedCompileCache("bfp8_grouped")
        ...
        compiled = _COMPILE_CACHE.get(compile_key)
        if compiled is None:
            compiled = cute.compile(...)
            _COMPILE_CACHE[compile_key] = compiled
    """

    def __init__(self, name: str):
        self.name = name
        self._mem: dict = {}
        self._hits = 0
        self._misses = 0

    def get(self, key, default=None):
        if key in self._mem:
            self._hits += 1
            if _is_verbose():
                _log.info(f"[JIT] {self.name}: cache hit (total: {len(self._mem)} entries)")
            return self._mem[key]
        self._misses += 1
        if _is_verbose():
            _log.info(f"[JIT] {self.name}: cache miss — compiling...")
        return default

    def __setitem__(self, key, value):
        global _compile_count, _compile_start_time
        _compile_count += 1
        if _compile_start_time is None:
            _compile_start_time = time.perf_counter()

        self._mem[key] = value

        if _is_verbose():
            elapsed = time.perf_counter() - _compile_start_time
            _log.info(
                f"[JIT] {self.name}: compiled "
                f"({_compile_count}/~{_EXPECTED_KERNEL_COUNT} kernels, "
                f"{elapsed:.1f}s elapsed)"
            )

    def __getitem__(self, key):
        return self._mem[key]

    def __contains__(self, key):
        hit = key in self._mem
        if hit:
            self._hits += 1
            if _is_verbose():
                _log.info(f"[JIT] {self.name}: cache hit")
        else:
            self._misses += 1
            if _is_verbose():
                _log.info(f"[JIT] {self.name}: cache miss — compiling...")
        return hit

    def __delitem__(self, key):
        del self._mem[key]

    def clear(self):
        self._mem.clear()

    def stats(self) -> dict:
        return {
            "name": self.name,
            "entries": len(self._mem),
            "hits": self._hits,
            "misses": self._misses,
        }

    def __repr__(self):
        return f"InstrumentedCompileCache({self.name!r}, entries={len(self._mem)})"


def cache_stats() -> dict:
    """Return disk cache statistics."""
    root = get_cache_root()
    stats = {}
    for sub in ("triton", "quack", "sonicmoe"):
        p = root / sub
        if p.exists():
            files = list(p.rglob("*"))
            stats[sub] = {
                "file_count": sum(1 for f in files if f.is_file()),
                "total_bytes": sum(f.stat().st_size for f in files if f.is_file()),
            }
    return stats


def clear_all_caches():
    """Clear all JIT caches (disk only — in-memory caches are per-process)."""
    import shutil

    root = get_cache_root()
    for sub in ("triton", "quack", "sonicmoe"):
        p = root / sub
        if p.exists():
            shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    if _is_verbose():
        _log.info(f"[JIT] Cleared all caches under {root}")
