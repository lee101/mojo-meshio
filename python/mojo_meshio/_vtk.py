from __future__ import annotations

from pathlib import Path

import numpy as np

from ._formats import ReadError, WriteError
from ._mesh import CellBlock, Mesh

_VTK_TO_CELL = {
    1: "vertex",
    3: "line",
    5: "triangle",
    7: "polygon",
    9: "quad",
    10: "tetra",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}
_CELL_TO_VTK = {value: key for key, value in _VTK_TO_CELL.items()}
_DTYPES = {
    "char": "i1",
    "unsigned_char": "u1",
    "short": "i2",
    "unsigned_short": "u2",
    "int": "i4",
    "unsigned_int": "u4",
    "long": "i8",
    "unsigned_long": "u8",
    "vtktypeint64": "i8",
    "vtktypeuint64": "u8",
    "float": "f4",
    "double": "f8",
}


def _dtype(name: str, endian: str = "=") -> np.dtype:
    try:
        return np.dtype(endian + _DTYPES[name.lower()])
    except KeyError as exc:
        raise ReadError(f"unsupported VTK scalar type {name!r}") from exc


def _cell_blocks(rows: list[np.ndarray], types: np.ndarray):
    blocks = []
    slices = []
    start = 0
    while start < len(rows):
        code = int(types[start])
        end = start + 1
        width = len(rows[start])
        while end < len(rows) and int(types[end]) == code and len(rows[end]) == width:
            end += 1
        name = _VTK_TO_CELL.get(code)
        if name is None:
            raise ReadError(f"unsupported VTK cell type code {code}")
        if name == "polygon":
            name = f"polygon{width}"
        data = np.asarray(rows[start:end], dtype=np.int64)
        if name == "wedge":
            data = data[:, [0, 2, 1, 3, 5, 4]]
        blocks.append(CellBlock(name, data))
        slices.append(slice(start, end))
        start = end
    return blocks, slices


def _numbers(lines: list[str], index: int, count: int, dtype) -> tuple[np.ndarray, int]:
    values = []
    while len(values) < count and index < len(lines):
        line = lines[index].strip()
        if line:
            values.extend(line.split())
        index += 1
    if len(values) != count:
        raise ReadError("truncated VTK numeric array")
    return np.asarray(values, dtype=dtype), index


def _read_ascii(raw: bytes) -> Mesh:
    lines = raw.decode("ascii").replace("\r", "").splitlines()
    if len(lines) < 4 or lines[3].strip() != "DATASET UNSTRUCTURED_GRID":
        raise ReadError("only legacy VTK UNSTRUCTURED_GRID is supported")
    index = 4
    points = None
    rows = []
    types = None
    point_data = {}
    flat_cell_data = {}
    data_target = None
    data_count = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        fields = line.split()
        tag = fields[0].upper()
        if tag == "POINTS":
            count = int(fields[1])
            values, index = _numbers(lines, index, count * 3, _dtype(fields[2]))
            points = values.reshape(count, 3)
        elif tag == "CELLS":
            first, second = int(fields[1]), int(fields[2])
            if index < len(lines) and lines[index].strip().upper().startswith("OFFSETS"):
                index += 1
                offsets, index = _numbers(lines, index, first, np.int64)
                if index >= len(lines) or not lines[index].strip().upper().startswith("CONNECTIVITY"):
                    raise ReadError("VTK 5 CELLS is missing CONNECTIVITY")
                index += 1
                connectivity, index = _numbers(lines, index, second, np.int64)
                rows = [
                    connectivity[offsets[i] : offsets[i + 1]]
                    for i in range(len(offsets) - 1)
                ]
            else:
                values, index = _numbers(lines, index, second, np.int64)
                cursor = 0
                rows = []
                for _ in range(first):
                    width = int(values[cursor])
                    rows.append(values[cursor + 1 : cursor + 1 + width])
                    cursor += width + 1
        elif tag == "CELL_TYPES":
            types, index = _numbers(lines, index, int(fields[1]), np.int64)
        elif tag == "POINT_DATA":
            data_target, data_count = point_data, int(fields[1])
        elif tag == "CELL_DATA":
            data_target, data_count = flat_cell_data, int(fields[1])
        elif tag == "FIELD":
            field_count = int(fields[2])
            if data_target is None:
                data_target = point_data
            for _ in range(field_count):
                header = lines[index].split()
                index += 1
                name, components, tuples, scalar_type = (
                    header[0],
                    int(header[1]),
                    int(header[2]),
                    header[3],
                )
                if data_count and tuples != data_count:
                    raise ReadError("VTK FIELD tuple count does not match data section")
                values, index = _numbers(
                    lines, index, components * tuples, _dtype(scalar_type)
                )
                data_target[name] = values if components == 1 else values.reshape(
                    tuples, components
                )
        else:
            raise ReadError(f"unsupported VTK section {tag!r}")
    if points is None or types is None or len(rows) != len(types):
        raise ReadError("incomplete VTK unstructured grid")
    blocks, slices = _cell_blocks(rows, types)
    cell_data = {
        name: [array[cell_slice] for cell_slice in slices]
        for name, array in flat_cell_data.items()
    }
    return Mesh(points, blocks, point_data=point_data, cell_data=cell_data)


def _line(raw: bytes, cursor: int) -> tuple[str, int]:
    end = raw.find(b"\n", cursor)
    if end < 0:
        return raw[cursor:].decode("ascii").strip(), len(raw)
    return raw[cursor:end].decode("ascii").strip(), end + 1


def _binary_array(raw: bytes, cursor: int, count: int, scalar_type: str):
    dtype = _dtype(scalar_type, ">")
    end = cursor + count * dtype.itemsize
    if end > len(raw):
        raise ReadError("truncated binary VTK array")
    array = np.frombuffer(raw, dtype=dtype, count=count, offset=cursor)
    cursor = end
    if cursor < len(raw) and raw[cursor] == 10:
        cursor += 1
    return array, cursor


def _read_binary(raw: bytes) -> Mesh:
    cursor = 0
    headers = []
    for _ in range(4):
        line, cursor = _line(raw, cursor)
        headers.append(line)
    if headers[3] != "DATASET UNSTRUCTURED_GRID":
        raise ReadError("only legacy VTK UNSTRUCTURED_GRID is supported")
    vtk5 = "Version 5." in headers[0]
    points = None
    rows = []
    types = None
    point_data = {}
    flat_cell_data = {}
    data_target = None
    data_count = 0
    while cursor < len(raw):
        line, cursor = _line(raw, cursor)
        if not line:
            continue
        fields = line.split()
        tag = fields[0].upper()
        if tag == "POINTS":
            count = int(fields[1])
            values, cursor = _binary_array(raw, cursor, count * 3, fields[2])
            points = values.reshape(count, 3)
        elif tag == "CELLS":
            first, second = int(fields[1]), int(fields[2])
            if vtk5:
                next_line, cursor = _line(raw, cursor)
                if not next_line.upper().startswith("OFFSETS"):
                    raise ReadError("VTK 5 CELLS is missing OFFSETS")
                scalar_type = next_line.split()[1]
                offsets, cursor = _binary_array(raw, cursor, first, scalar_type)
                connect_line, cursor = _line(raw, cursor)
                if not connect_line.upper().startswith("CONNECTIVITY"):
                    raise ReadError("VTK 5 CELLS is missing CONNECTIVITY")
                scalar_type = connect_line.split()[1]
                connectivity, cursor = _binary_array(raw, cursor, second, scalar_type)
                rows = [
                    connectivity[int(offsets[i]) : int(offsets[i + 1])]
                    for i in range(len(offsets) - 1)
                ]
            else:
                values, cursor = _binary_array(raw, cursor, second, "int")
                position = 0
                rows = []
                for _ in range(first):
                    width = int(values[position])
                    rows.append(values[position + 1 : position + 1 + width])
                    position += width + 1
        elif tag == "CELL_TYPES":
            types, cursor = _binary_array(raw, cursor, int(fields[1]), "int")
        elif tag == "POINT_DATA":
            data_target, data_count = point_data, int(fields[1])
        elif tag == "CELL_DATA":
            data_target, data_count = flat_cell_data, int(fields[1])
        elif tag == "FIELD":
            for _ in range(int(fields[2])):
                header, cursor = _line(raw, cursor)
                parts = header.split()
                name, components, tuples, scalar_type = (
                    parts[0],
                    int(parts[1]),
                    int(parts[2]),
                    parts[3],
                )
                if data_count and tuples != data_count:
                    raise ReadError("VTK FIELD tuple count does not match data section")
                values, cursor = _binary_array(
                    raw, cursor, components * tuples, scalar_type
                )
                data_target[name] = values if components == 1 else values.reshape(
                    tuples, components
                )
        else:
            raise ReadError(f"unsupported binary VTK section {tag!r}")
    if points is None or types is None or len(rows) != len(types):
        raise ReadError("incomplete VTK unstructured grid")
    blocks, slices = _cell_blocks(rows, types)
    cell_data = {
        name: [array[cell_slice] for cell_slice in slices]
        for name, array in flat_cell_data.items()
    }
    return Mesh(points, blocks, point_data=point_data, cell_data=cell_data)


def read_vtk(filename) -> Mesh:
    raw = Path(filename).read_bytes()
    first_lines = raw.split(b"\n", 4)
    if len(first_lines) < 4 or not first_lines[0].startswith(b"# vtk DataFile Version"):
        raise ReadError("expected a legacy VTK header")
    encoding = first_lines[2].strip()
    if encoding == b"ASCII":
        return _read_ascii(raw)
    if encoding == b"BINARY":
        return _read_binary(raw)
    raise ReadError(f"unsupported VTK encoding {encoding!r}")


def _vtk_type(array: np.ndarray) -> str:
    kind = array.dtype.kind
    if kind == "f":
        return "float" if array.dtype.itemsize <= 4 else "double"
    if kind == "u":
        return "unsigned_int" if array.dtype.itemsize <= 4 else "vtktypeuint64"
    if kind in "ib":
        return "int" if array.dtype.itemsize <= 4 else "vtktypeint64"
    raise WriteError(f"VTK cannot write dtype {array.dtype}")


def _write_values(stream, array: np.ndarray, scalar_type: str, binary: bool) -> None:
    flat = np.asarray(array).reshape(-1)
    if binary:
        dtype = _dtype(scalar_type, ">")
        stream.write(np.asarray(flat, dtype=dtype).tobytes())
        stream.write(b"\n")
    else:
        text = " ".join(str(value) for value in flat.tolist()) + "\n"
        stream.write(text.encode("ascii"))


def _write_field(stream, data: dict, count: int, binary: bool) -> None:
    arrays = {
        name: np.asarray(value)
        for name, value in data.items()
        if np.asarray(value).ndim in (1, 2) and len(value) == count
    }
    stream.write(f"FIELD FieldData {len(arrays)}\n".encode("ascii"))
    for name, array in arrays.items():
        components = 1 if array.ndim == 1 else array.shape[1]
        scalar_type = _vtk_type(array)
        stream.write(
            f"{name} {components} {count} {scalar_type}\n".encode("ascii")
        )
        _write_values(stream, array, scalar_type, binary)


def write_vtk(filename, mesh: Mesh, binary: bool = True) -> None:
    points = np.asarray(mesh.points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    rows = []
    cell_types = []
    for block in mesh.cells:
        base = "polygon" if block.type.startswith("polygon") else block.type
        if base not in _CELL_TO_VTK:
            raise WriteError(f"VTK writer does not support {block.type!r}")
        for row in np.asarray(block.data):
            row = np.asarray(row, dtype=np.int64)
            if base == "wedge":
                row = row[[0, 2, 1, 3, 5, 4]]
            rows.append(row)
            cell_types.append(_CELL_TO_VTK[base])
    connectivity = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    offsets = np.empty(len(rows) + 1, dtype=np.int64)
    offsets[0] = 0
    if rows:
        np.cumsum([len(row) for row in rows], out=offsets[1:])
    with open(filename, "wb") as stream:
        stream.write(b"# vtk DataFile Version 5.1\n")
        stream.write(b"written by mojo-meshio\n")
        stream.write(b"BINARY\n" if binary else b"ASCII\n")
        stream.write(b"DATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {len(points)} double\n".encode("ascii"))
        _write_values(stream, points, "double", binary)
        stream.write(f"CELLS {len(offsets)} {len(connectivity)}\n".encode("ascii"))
        stream.write(b"OFFSETS vtktypeint64\n")
        _write_values(stream, offsets, "vtktypeint64", binary)
        stream.write(b"CONNECTIVITY vtktypeint64\n")
        _write_values(stream, connectivity, "vtktypeint64", binary)
        stream.write(f"CELL_TYPES {len(rows)}\n".encode("ascii"))
        _write_values(stream, np.asarray(cell_types), "int", binary)
        if mesh.point_data:
            stream.write(f"POINT_DATA {len(points)}\n".encode("ascii"))
            _write_field(stream, mesh.point_data, len(points), binary)
        if mesh.cell_data:
            flattened = {}
            for name, arrays in mesh.cell_data.items():
                if len(arrays) == len(mesh.cells):
                    flattened[name] = np.concatenate(arrays)
            stream.write(f"CELL_DATA {len(rows)}\n".encode("ascii"))
            _write_field(stream, flattened, len(rows), binary)
