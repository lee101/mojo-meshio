from __future__ import annotations

import base64
import json
import struct
from pathlib import Path

import numpy as np

from ._formats import ReadError, WriteError
from ._mesh import CellBlock, Mesh

_COMPONENT_DTYPES = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_TYPE_WIDTH = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_CELL_MODE = {"vertex": 0, "line": 1, "triangle": 4}
_MODE_CELL = {value: key for key, value in _CELL_MODE.items()}


def _load(filename):
    path = Path(filename)
    raw = path.read_bytes()
    binary_chunk = None
    if path.suffix.lower() == ".glb":
        if len(raw) < 20 or raw[:4] != b"glTF":
            raise ReadError("invalid GLB header")
        magic, version, total = struct.unpack_from("<4sII", raw)
        if version != 2 or total != len(raw):
            raise ReadError("only GLB 2.0 is supported")
        cursor = 12
        document = None
        while cursor + 8 <= len(raw):
            length, chunk_type = struct.unpack_from("<II", raw, cursor)
            cursor += 8
            if length % 4 or cursor + length > len(raw):
                raise ReadError("invalid GLB chunk length")
            chunk = raw[cursor : cursor + length]
            cursor += length
            if chunk_type == 0x4E4F534A:
                document = json.loads(chunk.rstrip(b" \0"))
            elif chunk_type == 0x004E4942:
                binary_chunk = chunk
        if document is None:
            raise ReadError("GLB has no JSON chunk")
    else:
        document = json.loads(raw)
    buffers = []
    for index, spec in enumerate(document.get("buffers", [])):
        uri = spec.get("uri")
        if uri is None:
            if index != 0 or binary_chunk is None:
                raise ReadError("GLTF buffer has no URI or GLB chunk")
            payload = binary_chunk
        elif uri.startswith("data:"):
            try:
                payload = base64.b64decode(uri.split(",", 1)[1], validate=True)
            except (ValueError, IndexError) as exc:
                raise ReadError("invalid GLTF data URI") from exc
        else:
            payload = (path.parent / uri).read_bytes()
        if len(payload) < spec["byteLength"]:
            raise ReadError("truncated GLTF buffer")
        buffers.append(payload)
    return document, buffers


def _accessor(document, buffers, index: int) -> np.ndarray:
    accessor = document["accessors"][index]
    if "sparse" in accessor:
        raise ReadError("sparse GLTF accessors are unsupported")
    view = document["bufferViews"][accessor["bufferView"]]
    dtype = _COMPONENT_DTYPES.get(accessor["componentType"])
    if dtype is None:
        raise ReadError("unsupported GLTF accessor component type")
    try:
        width = _TYPE_WIDTH[accessor["type"]]
    except KeyError as exc:
        raise ReadError("unsupported GLTF accessor type") from exc
    count = accessor["count"]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", dtype.itemsize * width)
    if count < 0 or offset < 0 or stride < dtype.itemsize * width:
        raise ReadError("invalid GLTF accessor layout")
    try:
        payload = buffers[view["buffer"]]
    except (IndexError, KeyError) as exc:
        raise ReadError("GLTF accessor references an invalid buffer") from exc
    required = offset + (count - 1) * stride + dtype.itemsize * width if count else offset
    view_start = view.get("byteOffset", 0)
    view_end = view_start + view["byteLength"]
    if required > len(payload) or required > view_end:
        raise ReadError("GLTF accessor exceeds its buffer")
    if stride == dtype.itemsize * width:
        result = np.frombuffer(payload, dtype=dtype, count=count * width, offset=offset)
        return result if width == 1 else result.reshape(count, width)
    result = np.ndarray(
        (count, width),
        dtype=dtype,
        buffer=payload,
        offset=offset,
        strides=(stride, dtype.itemsize),
    )
    return result[:, 0] if width == 1 else result


def read_gltf(filename) -> Mesh:
    document, buffers = _load(filename)
    meshes = document.get("meshes", [])
    if not meshes:
        raise ReadError("GLTF contains no meshes")
    primitives = meshes[0].get("primitives", [])
    if not primitives:
        raise ReadError("GLTF mesh contains no primitives")
    position_accessors = []
    for primitive in primitives:
        try:
            position_accessors.append(primitive["attributes"]["POSITION"])
        except KeyError as exc:
            raise ReadError("GLTF primitive has no POSITION accessor") from exc
    unique_positions = list(dict.fromkeys(position_accessors))
    point_arrays = [_accessor(document, buffers, index) for index in unique_positions]
    if any(array.ndim != 2 or array.shape[1] != 3 for array in point_arrays):
        raise ReadError("GLTF POSITION must be a VEC3 accessor")
    bases = {}
    offset = 0
    for accessor_index, array in zip(unique_positions, point_arrays):
        bases[accessor_index] = offset
        offset += len(array)
    points = point_arrays[0] if len(point_arrays) == 1 else np.concatenate(point_arrays)
    blocks = []
    for primitive, position_index in zip(primitives, position_accessors):
        mode = primitive.get("mode", 4)
        cell_type = _MODE_CELL.get(mode)
        if cell_type is None:
            raise ReadError(f"unsupported GLTF primitive mode {mode}")
        width = {"vertex": 1, "line": 2, "triangle": 3}[cell_type]
        if "indices" in primitive:
            indices = _accessor(document, buffers, primitive["indices"])
        else:
            indices = np.arange(len(point_arrays[unique_positions.index(position_index)]))
        if indices.ndim != 1 or len(indices) % width:
            raise ReadError("GLTF index accessor has an invalid size")
        if indices.dtype.kind not in "iu":
            raise ReadError("GLTF indices must use an integer component type")
        point_count = len(point_arrays[unique_positions.index(position_index)])
        if indices.size and int(indices.max()) >= point_count:
            raise ReadError("GLTF index is out of bounds")
        data = indices.reshape(-1, width)
        base = bases[position_index]
        if base:
            data = data.astype(np.int64) + base
        blocks.append(CellBlock(cell_type, data))
    return Mesh(points, blocks)


def _pad(blob: bytearray, alignment: int = 4) -> None:
    blob.extend(b"\0" * ((-len(blob)) % alignment))


def write_gltf(filename, mesh: Mesh, binary: bool = False) -> None:
    points = np.asarray(mesh.points)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    points = np.asarray(points[:, :3], dtype="<f4", order="C")
    blob = bytearray()
    views = []
    accessors = []

    def add(array: np.ndarray, component: int, kind: str, target: int) -> int:
        _pad(blob)
        offset = len(blob)
        blob.extend(array.tobytes(order="C"))
        view_index = len(views)
        views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": array.nbytes,
                "target": target,
            }
        )
        width = _TYPE_WIDTH[kind]
        count = array.size // width
        accessor = {
            "bufferView": view_index,
            "componentType": component,
            "count": count,
            "type": kind,
        }
        if kind == "VEC3" and count:
            shaped = array.reshape(count, 3)
            accessor["min"] = shaped.min(axis=0).astype(float).tolist()
            accessor["max"] = shaped.max(axis=0).astype(float).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    position_accessor = add(points, 5126, "VEC3", 34962)
    primitives = []
    for block in mesh.cells:
        if block.type not in _CELL_MODE:
            raise WriteError(f"GLTF writer does not support {block.type!r}")
        indices64 = np.asarray(block.data, dtype=np.int64)
        if len(indices64) and (indices64.min() < 0 or indices64.max() >= len(points)):
            raise WriteError("GLTF cell index is out of bounds")
        indices = np.asarray(indices64.reshape(-1), dtype="<u4")
        index_accessor = add(indices, 5125, "SCALAR", 34963)
        primitives.append(
            {
                "attributes": {"POSITION": position_accessor},
                "indices": index_accessor,
                "mode": _CELL_MODE[block.type],
            }
        )
    document = {
        "asset": {"version": "2.0", "generator": "mojo-meshio"},
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
        "meshes": [{"primitives": primitives}],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    path = Path(filename)
    if binary:
        json_bytes = json.dumps(document, separators=(",", ":")).encode("utf8")
        json_bytes += b" " * ((-len(json_bytes)) % 4)
        _pad(blob)
        total = 12 + 8 + len(json_bytes) + 8 + len(blob)
        with open(path, "wb") as stream:
            stream.write(struct.pack("<4sII", b"glTF", 2, total))
            stream.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
            stream.write(json_bytes)
            stream.write(struct.pack("<II", len(blob), 0x004E4942))
            stream.write(blob)
    else:
        document["buffers"][0]["uri"] = (
            "data:application/octet-stream;base64,"
            + base64.b64encode(blob).decode("ascii")
        )
        path.write_text(
            json.dumps(document, separators=(",", ":")),
            encoding="utf8",
        )
