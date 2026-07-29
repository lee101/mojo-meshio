from __future__ import annotations

import numpy as np

from ._lib import address, as_f64_c, as_i64_c, ensure_parallel_runtime, lib

_GATHER_PARALLEL_THRESHOLD = 16384


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
    if len(c) >= _GATHER_PARALLEL_THRESHOLD:
        ensure_parallel_runtime()
    lib().mmi_gather_triangles(address(p), address(c), address(result), len(c))
    return result
