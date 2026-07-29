from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ._formats import ReadError, WriteError
from ._geometry import gather_triangles, triangle_normals, weld_triangles
from ._mesh import Mesh


def _binary_layout(raw: bytes) -> tuple[bool, int]:
    if len(raw) < 84:
        return False, 0
    count = struct.unpack_from("<I", raw, 80)[0]
    return 84 + count * 50 == len(raw), count


def _read_binary(raw: bytes, count: int) -> Mesh:
    dtype = np.dtype(
        [
            ("normal", "<f4", (3,)),
            ("vertices", "<f4", (3, 3)),
            ("attribute", "<u2"),
        ]
    )
    facets = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    vertices = np.asarray(facets["vertices"], dtype=np.float64)
    points, cells = weld_triangles(vertices)
    normals = np.asarray(facets["normal"], dtype=np.float64)
    return Mesh(
        points,
        [("triangle", cells)],
        cell_data={"facet_normals": [normals]},
    )


def _read_ascii(raw: bytes) -> Mesh:
    vertices = []
    normals = []
    current = []
    for line_number, raw_line in enumerate(raw.decode("ascii").splitlines(), 1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        try:
            if fields[0].lower() == "facet" and len(fields) >= 5:
                normals.append([float(value) for value in fields[-3:]])
            elif fields[0].lower() == "vertex" and len(fields) == 4:
                current.append([float(value) for value in fields[1:]])
                if len(current) == 3:
                    vertices.append(current)
                    current = []
        except ValueError as exc:
            raise ReadError(f"invalid ASCII STL number at line {line_number}") from exc
    if current or not vertices:
        raise ReadError("ASCII STL has incomplete or no facets")
    triangles = np.asarray(vertices, dtype=np.float64)
    points, cells = weld_triangles(triangles)
    normal_array = (
        np.asarray(normals, dtype=np.float64)
        if len(normals) == len(triangles)
        else triangle_normals(points, cells)
    )
    return Mesh(
        points,
        [("triangle", cells)],
        cell_data={"facet_normals": [normal_array]},
    )


def read_stl(filename) -> Mesh:
    raw = Path(filename).read_bytes()
    binary, count = _binary_layout(raw)
    return _read_binary(raw, count) if binary else _read_ascii(raw)


def _triangles(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    normal_arrays = []
    supplied = mesh.cell_data.get("facet_normals", [])
    for index, block in enumerate(mesh.cells):
        if block.type != "triangle":
            if len(block.data):
                raise WriteError("STL supports triangle cells only")
            continue
        arrays.append(np.asarray(block.data))
        if index < len(supplied) and len(supplied[index]) == len(block.data):
            normal_arrays.append(np.asarray(supplied[index], dtype=np.float64))
        else:
            normal_arrays.append(triangle_normals(mesh.points, block.data))
    if not arrays:
        raise WriteError("STL requires at least one triangle")
    return np.concatenate(arrays), np.concatenate(normal_arrays)


def write_stl(filename, mesh: Mesh, binary: bool = False) -> None:
    cells, normals = _triangles(mesh)
    vertices = gather_triangles(mesh.points, cells)
    if binary:
        header = b"mojo-meshio binary STL".ljust(80, b"\0")
        dtype = np.dtype(
            [
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (3, 3)),
                ("attribute", "<u2"),
            ]
        )
        facets = np.zeros(len(cells), dtype=dtype)
        facets["normal"] = normals
        facets["vertices"] = vertices
        with open(filename, "wb") as stream:
            stream.write(header)
            stream.write(struct.pack("<I", len(facets)))
            stream.write(facets.tobytes())
    else:
        with open(filename, "w", encoding="ascii", newline="\n") as stream:
            stream.write("solid\n")
            for normal, triangle in zip(normals, vertices):
                stream.write(
                    "facet normal " + " ".join(f"{value:.17g}" for value in normal) + "\n"
                )
                stream.write("  outer loop\n")
                for vertex in triangle:
                    stream.write(
                        "    vertex "
                        + " ".join(f"{value:.17g}" for value in vertex)
                        + "\n"
                    )
                stream.write("  endloop\nendfacet\n")
            stream.write("endsolid\n")
