from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ._formats import ReadError, WriteError
from ._lib import scan_f64
from ._mesh import CellBlock, Mesh


def _blocks(grouped: dict[int, list[list[int]]]) -> list[CellBlock]:
    names = {1: "vertex", 2: "line", 3: "triangle", 4: "quad"}
    return [
        CellBlock(names.get(width, f"polygon{width}"), np.asarray(rows, dtype=np.int64))
        for width, rows in grouped.items()
    ]


def read_obj(filename) -> Mesh:
    points: list[list[float]] = []
    grouped: dict[int, list[list[int]]] = defaultdict(list)
    texcoords: list[list[float]] = []
    normals: list[list[float]] = []
    with open(filename, "r", encoding="utf8", errors="strict") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.partition("#")[0].strip()
            if not line:
                continue
            fields = line.split()
            tag = fields[0]
            try:
                if tag == "v":
                    values = [float(value) for value in fields[1:]]
                    if len(values) < 3:
                        raise ValueError("vertex has fewer than three coordinates")
                    points.append(values[:3])
                elif tag == "vt":
                    texcoords.append([float(value) for value in fields[1:]])
                elif tag == "vn":
                    normals.append([float(value) for value in fields[1:4]])
                elif tag in ("f", "l", "p"):
                    row = []
                    for token in fields[1:]:
                        raw_index = int(token.split("/", 1)[0])
                        index = raw_index - 1 if raw_index > 0 else len(points) + raw_index
                        if index < 0 or index >= len(points):
                            raise ValueError(f"vertex index {raw_index} is out of bounds")
                        row.append(index)
                    if tag == "l" and len(row) > 2:
                        for left, right in zip(row, row[1:]):
                            grouped[2].append([left, right])
                    else:
                        grouped[len(row)].append(row)
            except ValueError as exc:
                raise ReadError(f"{filename}:{line_number}: {exc}") from exc
    if not points:
        raise ReadError(f"{filename}: no vertices found")
    point_data = {}
    if len(normals) == len(points):
        point_data["obj:vn"] = np.asarray(normals, dtype=np.float64)
    if len(texcoords) == len(points):
        point_data["obj:vt"] = np.asarray(texcoords, dtype=np.float64)
    return Mesh(np.asarray(points, dtype=np.float64), _blocks(grouped), point_data=point_data)


def write_obj(filename, mesh: Mesh) -> None:
    points = np.asarray(mesh.points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    normals = mesh.point_data.get("obj:vn")
    texcoords = mesh.point_data.get("obj:vt")
    with open(filename, "w", encoding="utf8", newline="\n") as stream:
        stream.write("# Created by mojo-meshio\n")
        np.savetxt(stream, points[:, :3], fmt="v %.17g %.17g %.17g")
        if texcoords is not None and len(texcoords) == len(points):
            for row in np.asarray(texcoords):
                stream.write("vt " + " ".join(f"{value:.17g}" for value in row) + "\n")
        if normals is not None and len(normals) == len(points):
            np.savetxt(stream, np.asarray(normals)[:, :3], fmt="vn %.17g %.17g %.17g")
        for block in mesh.cells:
            if block.type == "vertex":
                prefix = "p"
            elif block.type == "line":
                prefix = "l"
            elif block.type in ("triangle", "quad") or block.type.startswith("polygon"):
                prefix = "f"
            else:
                raise WriteError(f"OBJ does not support {block.type!r} cells")
            for row in np.asarray(block.data):
                stream.write(prefix + " " + " ".join(str(int(i) + 1) for i in row) + "\n")


def read_off(filename) -> Mesh:
    raw = Path(filename).read_bytes()
    payload_start = None
    cursor = 0
    while cursor < len(raw):
        line_end = raw.find(b"\n", cursor)
        if line_end < 0:
            line_end = len(raw)
            next_line = len(raw)
        else:
            next_line = line_end + 1
        line = raw[cursor:line_end].partition(b"#")[0].strip()
        if line:
            if line != b"OFF":
                break
            payload_start = next_line
            break
        cursor = next_line
    if payload_start is None:
        raise ReadError(f"{filename}: expected OFF header")
    payload = memoryview(raw)[payload_start:]
    values = scan_f64(payload, max(4, len(payload) // 2 + 1))
    if len(values) < 3:
        raise ReadError(f"{filename}: missing OFF counts")
    point_count, face_count = int(values[0]), int(values[1])
    cursor = 3
    needed = cursor + point_count * 3
    if point_count < 0 or face_count < 0 or len(values) < needed:
        raise ReadError(f"{filename}: truncated OFF point table")
    points = values[cursor:needed].reshape(point_count, 3).copy()
    cursor = needed
    if face_count:
        width = int(values[cursor]) if cursor < len(values) else 0
        record_width = width + 1
        face_end = cursor + face_count * record_width
        if width >= 1 and face_end <= len(values):
            records = values[cursor:face_end].reshape(face_count, record_width)
            if np.all(records[:, 0] == width):
                cells = records[:, 1:].astype(np.int64)
                if np.any(cells < 0) or np.any(cells >= point_count):
                    raise ReadError(f"{filename}: OFF face index out of bounds")
                name = {1: "vertex", 2: "line", 3: "triangle", 4: "quad"}.get(
                    width, f"polygon{width}"
                )
                return Mesh(points, [CellBlock(name, cells)])
    grouped: dict[int, list[list[int]]] = defaultdict(list)
    for _ in range(face_count):
        if cursor >= len(values):
            raise ReadError(f"{filename}: truncated OFF face table")
        width = int(values[cursor])
        cursor += 1
        if width < 1 or cursor + width > len(values):
            raise ReadError(f"{filename}: invalid OFF face width")
        row = values[cursor : cursor + width].astype(np.int64).tolist()
        if any(index < 0 or index >= point_count for index in row):
            raise ReadError(f"{filename}: OFF face index out of bounds")
        grouped[width].append(row)
        cursor += width
    return Mesh(points, _blocks(grouped))


def write_off(filename, mesh: Mesh) -> None:
    supported = [
        block
        for block in mesh.cells
        if block.type in ("triangle", "quad") or block.type.startswith("polygon")
    ]
    if len(supported) != len(mesh.cells):
        raise WriteError("OFF supports polygonal surface cells only")
    face_count = sum(len(block.data) for block in supported)
    points = np.asarray(mesh.points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    with open(filename, "w", encoding="ascii", newline="\n") as stream:
        stream.write(f"OFF\n{len(points)} {face_count} 0\n")
        np.savetxt(stream, points[:, :3], fmt="%.17g")
        for block in supported:
            width = block.data.shape[1]
            for row in block.data:
                stream.write(
                    str(width) + " " + " ".join(str(int(index)) for index in row) + "\n"
                )
