"""Benchmarks against meshio and NumPy on identical data."""

from __future__ import annotations

import gc
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import meshio
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_meshio as mm  # noqa: E402


def timeit(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        if result is None:
            raise RuntimeError("benchmark returned no result")
        best = min(best, elapsed)
    return best


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rng = np.random.default_rng(7)
    triangle_count = 500_000
    points = np.ascontiguousarray(rng.normal(size=(triangle_count * 3, 3)))
    cells = np.arange(triangle_count * 3, dtype=np.int64).reshape(-1, 3)

    def numpy_normals():
        triangles = points[cells]
        values = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        lengths = np.linalg.norm(values, axis=1)
        values[lengths != 0] /= lengths[lengths != 0, None]
        values[lengths == 0] = 0
        return values

    cases = [
        (
            "triangle normals (500k)",
            lambda: mm.triangle_normals(points, cells),
            numpy_normals,
            "NumPy",
        ),
        (
            "indexed triangle gather (500k)",
            lambda: mm.gather_triangles(points, cells),
            lambda: points[cells],
            "NumPy",
        ),
    ]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        io_count = 150_000
        io_points = points[: io_count * 3]
        io_cells = cells[:io_count]
        io_mesh = mm.Mesh(io_points, [("triangle", io_cells)])
        stl_path = root / "large.stl"
        off_path = root / "large.off"
        mm.write(stl_path, io_mesh, binary=True)
        mm.write(off_path, io_mesh)
        cases.extend(
            [
                (
                    "binary STL read (150k triangles)",
                    lambda: mm.read(stl_path),
                    lambda: meshio.read(stl_path),
                    "meshio",
                ),
                (
                    "ASCII OFF read (150k triangles)",
                    lambda: mm.read(off_path),
                    lambda: meshio.read(off_path),
                    "meshio",
                ),
            ]
        )

        print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}")
        print()
        print("| benchmark | mojo-meshio | reference | relative |")
        print("| --- | ---: | ---: | ---: |")
        for name, ours, reference, reference_name in cases:
            ours()
            reference()
            ours_time = timeit(ours)
            reference_time = timeit(reference)
            ratio = reference_time / ours_time
            label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
            print(
                f"| {name} | {ours_time * 1000:.2f} ms | "
                f"{reference_time * 1000:.2f} ms ({reference_name}) | {label} |"
            )


if __name__ == "__main__":
    main()
