from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ._lib import address, as_f64_c, as_i64_c, lib

_GATHER_PARALLEL_THRESHOLD = 16384
_GATHER_WORKERS = min(os.cpu_count() or 1, 16)
_gather_executor: ThreadPoolExecutor | None = None


def _executor() -> ThreadPoolExecutor:
    global _gather_executor
    if _gather_executor is None:
        _gather_executor = ThreadPoolExecutor(max_workers=_GATHER_WORKERS)
    return _gather_executor


def _geometry_inputs(points, cells, operation: str) -> tuple[np.ndarray, np.ndarray]:
    p = as_f64_c(points)
    c = as_i64_c(cells)
    if p.ndim != 2 or p.shape[1] != 3 or c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"{operation} expects points (n, 3) and cells (m, 3)")
    if c.size and (c.min() < 0 or c.max() >= len(p)):
        raise IndexError(f"{operation} cell index is out of bounds")
    return p, c


def triangle_normals(points, cells) -> np.ndarray:
    p, c = _geometry_inputs(points, cells, "triangle_normals")
    storage = np.empty((max(len(c), 1), 3), dtype=np.float64)
    result = storage[: len(c)]
    lib().mmi_triangle_normals(address(p), address(c), address(result), len(c))
    return result


def weld_triangles(vertices) -> tuple[np.ndarray, np.ndarray]:
    v = as_f64_c(vertices)
    if v.ndim == 3 and v.shape[1:] == (3, 3):
        triangles = len(v)
        v = v.reshape(-1, 3)
    elif v.ndim == 2 and v.shape[1] == 3 and len(v) % 3 == 0:
        triangles = len(v) // 3
    else:
        raise ValueError("vertices must have shape (n, 3, 3) or (3*n, 3)")
    maximum = len(v)
    capacity = 1
    while capacity < maximum * 2:
        capacity <<= 1
    point_storage = np.empty((max(maximum, 1), 3), dtype=np.float64)
    cell_storage = np.empty((max(triangles, 1), 3), dtype=np.int64)
    points = point_storage[:maximum]
    cells = cell_storage[:triangles]
    table = np.full(max(capacity, 1), -1, dtype=np.int64)
    count = lib().mmi_weld_triangles(
        address(v), triangles, address(points), address(cells), address(table), len(table)
    )
    return points[:count], cells


def gather_triangles(points, cells) -> np.ndarray:
    p, c = _geometry_inputs(points, cells, "gather_triangles")
    storage = np.empty((max(len(c), 1), 3, 3), dtype=np.float64)
    result = storage[: len(c)]
    native = lib()
    p_address = address(p)
    c_address = address(c)
    result_address = address(result)
    if len(c) < _GATHER_PARALLEL_THRESHOLD or _GATHER_WORKERS == 1:
        native.mmi_gather_triangles(p_address, c_address, result_address, len(c))
        return result

    chunk_size = (len(c) + _GATHER_WORKERS - 1) // _GATHER_WORKERS
    futures = []
    for start in range(0, len(c), chunk_size):
        stop = min(start + chunk_size, len(c))
        futures.append(
            _executor().submit(
                native.mmi_gather_triangles_range,
                p_address,
                c_address,
                result_address,
                start,
                stop,
            )
        )
    for future in futures:
        future.result()
    return result
