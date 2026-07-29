from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._formats import ReadError, WriteError
from ._lib import scan_f64
from ._mesh import Mesh
from ._text import _blocks

_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _header(raw: bytes):
    end = raw.find(b"end_header")
    if end < 0:
        raise ReadError("PLY header has no end_header")
    newline = raw.find(b"\n", end)
    data_start = len(raw) if newline < 0 else newline + 1
    lines = raw[:data_start].decode("ascii").replace("\r", "").splitlines()
    if not lines or lines[0] != "ply":
        raise ReadError("expected PLY header")
    fmt = None
    elements = []
    current = None
    for line in lines[1:]:
        fields = line.split()
        if not fields or fields[0] in ("comment", "obj_info", "end_header"):
            continue
        if fields[0] == "format":
            fmt = fields[1]
        elif fields[0] == "element":
            current = {"name": fields[1], "count": int(fields[2]), "properties": []}
            elements.append(current)
        elif fields[0] == "property" and current is not None:
            if fields[1] == "list":
                current["properties"].append(("list", fields[2], fields[3], fields[4]))
            else:
                current["properties"].append(("scalar", fields[1], fields[2]))
    if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ReadError(f"unsupported PLY encoding {fmt!r}")
    return fmt, elements, data_start


def _read_ascii(body: bytes, elements) -> Mesh:
    values = scan_f64(body, max(4, len(body) // 2 + 1))
    cursor = 0
    points = None
    point_data = {}
    grouped: dict[int, list[list[int]]] = defaultdict(list)
    for element in elements:
        count = element["count"]
        properties = element["properties"]
        if element["name"] == "vertex":
            if any(prop[0] != "scalar" for prop in properties):
                raise ReadError("list properties on PLY vertices are unsupported")
            width = len(properties)
            end = cursor + count * width
            if end > len(values):
                raise ReadError("truncated ASCII PLY vertex table")
            table = values[cursor:end].reshape(count, width)
            names = [prop[2] for prop in properties]
            if not all(name in names for name in ("x", "y", "z")):
                raise ReadError("PLY vertices require x, y, and z properties")
            points = table[:, [names.index("x"), names.index("y"), names.index("z")]].copy()
            for column, name in enumerate(names):
                if name not in ("x", "y", "z"):
                    point_data[name] = table[:, column].copy()
            cursor = end
        elif element["name"] == "face":
            list_props = [prop for prop in properties if prop[0] == "list"]
            scalars = [prop for prop in properties if prop[0] == "scalar"]
            if len(list_props) != 1 or scalars:
                raise ReadError("PLY faces must contain one vertex_indices list")
            for _ in range(count):
                if cursor >= len(values):
                    raise ReadError("truncated ASCII PLY face table")
                width = int(values[cursor])
                cursor += 1
                if cursor + width > len(values):
                    raise ReadError("truncated ASCII PLY face")
                grouped[width].append(values[cursor : cursor + width].astype(np.int64).tolist())
                cursor += width
        else:
            scalar_width = sum(prop[0] == "scalar" for prop in properties)
            if scalar_width != len(properties):
                raise ReadError(f"unsupported list property on element {element['name']!r}")
            cursor += count * scalar_width
    if points is None:
        raise ReadError("PLY has no vertex element")
    return Mesh(points, _blocks(grouped), point_data=point_data)


def _scalar(stream: memoryview, cursor: int, dtype: str, endian: str):
    np_dtype = np.dtype(endian + _PLY_DTYPES[dtype])
    end = cursor + np_dtype.itemsize
    if end > len(stream):
        raise ReadError("truncated binary PLY data")
    return np.frombuffer(stream[cursor:end], dtype=np_dtype, count=1)[0], end


def _read_binary(body: memoryview, elements, endian: str) -> Mesh:
    cursor = 0
    points = None
    point_data = {}
    grouped: dict[int, list[list[int]]] = defaultdict(list)
    for element in elements:
        count = element["count"]
        properties = element["properties"]
        if element["name"] == "vertex" and all(prop[0] == "scalar" for prop in properties):
            dtype = np.dtype(
                [(prop[2], endian + _PLY_DTYPES[prop[1]]) for prop in properties]
            )
            end = cursor + count * dtype.itemsize
            if end > len(body):
                raise ReadError("truncated binary PLY vertex table")
            table = np.frombuffer(body[cursor:end], dtype=dtype, count=count)
            names = table.dtype.names or ()
            if not all(name in names for name in ("x", "y", "z")):
                raise ReadError("PLY vertices require x, y, and z properties")
            points = np.column_stack((table["x"], table["y"], table["z"]))
            point_data = {
                name: np.asarray(table[name]).copy()
                for name in names
                if name not in ("x", "y", "z")
            }
            cursor = end
        elif element["name"] == "face":
            if len(properties) != 1 or properties[0][0] != "list":
                raise ReadError("binary PLY faces must contain one list property")
            _, count_type, index_type, _ = properties[0]
            index_dtype = np.dtype(endian + _PLY_DTYPES[index_type])
            for _ in range(count):
                width_value, cursor = _scalar(body, cursor, count_type, endian)
                width = int(width_value)
                end = cursor + width * index_dtype.itemsize
                if end > len(body):
                    raise ReadError("truncated binary PLY face")
                row = np.frombuffer(body[cursor:end], dtype=index_dtype, count=width)
                grouped[width].append(row.astype(np.int64).tolist())
                cursor = end
        else:
            if any(prop[0] != "scalar" for prop in properties):
                raise ReadError(f"unsupported list property on element {element['name']!r}")
            width = sum(np.dtype(_PLY_DTYPES[prop[1]]).itemsize for prop in properties)
            cursor += count * width
    if points is None:
        raise ReadError("PLY has no vertex element")
    return Mesh(points, _blocks(grouped), point_data=point_data)


def read_ply(filename) -> Mesh:
    raw = Path(filename).read_bytes()
    fmt, elements, data_start = _header(raw)
    if fmt == "ascii":
        return _read_ascii(raw[data_start:], elements)
    endian = "<" if fmt == "binary_little_endian" else ">"
    return _read_binary(memoryview(raw)[data_start:], elements, endian)


def write_ply(filename, mesh: Mesh, binary: bool = True) -> None:
    points = np.asarray(mesh.points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    blocks = [
        block
        for block in mesh.cells
        if block.type in ("triangle", "quad") or block.type.startswith("polygon")
    ]
    if len(blocks) != len(mesh.cells):
        raise WriteError("PLY writer supports polygonal surface cells only")
    scalar_data = {
        name: np.asarray(array)
        for name, array in mesh.point_data.items()
        if np.asarray(array).ndim == 1 and len(array) == len(points)
    }
    face_count = sum(len(block.data) for block in blocks)
    if any(
        block.data.size
        and (block.data.min() < np.iinfo(np.int32).min or block.data.max() > np.iinfo(np.int32).max)
        for block in blocks
    ):
        raise WriteError("PLY face index cannot be represented as int32")
    encoding = "binary_little_endian" if binary else "ascii"
    header = [
        "ply",
        f"format {encoding} 1.0",
        "comment Created by mojo-meshio",
        f"element vertex {len(points)}",
        "property double x",
        "property double y",
        "property double z",
    ]
    for name in scalar_data:
        header.append(f"property double {name}")
    header.extend(
        [
            f"element face {face_count}",
            "property list uchar int vertex_indices",
            "end_header",
            "",
        ]
    )
    with open(filename, "wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        if binary:
            columns = [points[:, i] for i in range(3)] + list(scalar_data.values())
            table = np.empty((len(points), len(columns)), dtype="<f8")
            for column, values in enumerate(columns):
                table[:, column] = values
            stream.write(table.tobytes())
            for block in blocks:
                width = block.data.shape[1]
                if width > 255:
                    raise WriteError("PLY face width exceeds uchar list count")
                for row in np.asarray(block.data, dtype=np.int64):
                    stream.write(struct.pack("<B", width))
                    stream.write(np.asarray(row, dtype="<i4").tobytes())
        else:
            table = np.column_stack(
                [points[:, i] for i in range(3)] + list(scalar_data.values())
            )
            np.savetxt(stream, table, fmt="%.17g")
            for block in blocks:
                width = block.data.shape[1]
                for row in block.data:
                    text = str(width) + " " + " ".join(str(int(index)) for index in row)
                    stream.write((text + "\n").encode("ascii"))
