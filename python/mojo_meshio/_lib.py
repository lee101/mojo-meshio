"""ctypes bindings for the Mojo geometry kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = Path(os.environ.get("MOJO_MESHIO_LIB", ROOT / "dist/libmojo-meshio.so"))
I = ctypes.c_int64

_SIGNATURES = {
    "mmi_parse_f64": ([I, I, I, I], I),
    "mmi_parse_i64": ([I, I, I, I], I),
    "mmi_triangle_normals": ([I, I, I, I], I),
    "mmi_weld_triangles": ([I, I, I, I, I, I], I),
    "mmi_gather_triangles": ([I, I, I, I], None),
}
_library: ctypes.CDLL | None = None
_parallel_runtime = None


def build() -> Path:
    sources = list((ROOT / "src").glob("*.mojo"))
    if LIB_PATH.exists() and all(LIB_PATH.stat().st_mtime >= p.stat().st_mtime for p in sources):
        return LIB_PATH
    proc = subprocess.run(
        ["bash", str(ROOT / "build/build.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if proc.returncode or not LIB_PATH.exists():
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB_PATH


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def ensure_parallel_runtime() -> None:
    global _parallel_runtime
    if _parallel_runtime is None:
        runtime = lib().KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice
        runtime.argtypes = []
        runtime.restype = ctypes.c_void_p
        runtime()
        _parallel_runtime = runtime


def address(array: np.ndarray) -> int:
    """Return a live, non-null address for an ABI argument."""
    value = int(array.ctypes.data)
    if not value:
        raise ValueError("cannot pass a null NumPy buffer to Mojo")
    return value


def byte_array(data: bytes | bytearray | memoryview) -> np.ndarray:
    result = np.frombuffer(data, dtype=np.uint8)
    # Mojo pointers are non-nullable even when the byte count is zero.
    return result if result.size else np.empty(1, dtype=np.uint8)


def scan_f64(data: bytes | bytearray | memoryview, capacity: int) -> np.ndarray:
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    source = byte_array(data)
    source_size = memoryview(data).nbytes
    storage = np.empty(max(capacity, 1), dtype=np.float64)
    count = lib().mmi_parse_f64(
        address(source), source_size, address(storage), capacity
    )
    if count < 0:
        raise ValueError(f"invalid numeric ASCII stream (kernel status {count})")
    return storage[:count]


def scan_i64(data: bytes | bytearray | memoryview, capacity: int) -> np.ndarray:
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    source = byte_array(data)
    source_size = memoryview(data).nbytes
    storage = np.empty(max(capacity, 1), dtype=np.int64)
    count = lib().mmi_parse_i64(
        address(source), source_size, address(storage), capacity
    )
    if count < 0:
        raise ValueError(f"invalid integer ASCII stream (kernel status {count})")
    return storage[:count]


def as_f64_c(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64)


def as_i64_c(array) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind not in "iu":
        raise TypeError("connectivity must contain integers")
    if value.size:
        minimum = value.min()
        maximum = value.max()
        limits = np.iinfo(np.int64)
        if minimum < limits.min or maximum > limits.max:
            raise OverflowError("connectivity cannot be represented as int64")
    return np.ascontiguousarray(value, dtype=np.int64)
