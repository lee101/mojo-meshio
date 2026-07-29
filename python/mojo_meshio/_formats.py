from __future__ import annotations

from pathlib import Path

from ._mesh import Mesh


class ReadError(RuntimeError):
    pass


class WriteError(RuntimeError):
    pass


_EXTENSIONS = {
    ".obj": "obj",
    ".off": "off",
    ".ply": "ply",
    ".stl": "stl",
    ".vtk": "vtk",
    ".gltf": "gltf",
    ".glb": "glb",
}


def _format(filename, file_format: str | None) -> str:
    if file_format:
        fmt = file_format.lower()
        aliases = {"vtk42": "vtk", "vtk51": "vtk"}
        return aliases.get(fmt, fmt)
    suffix = Path(filename).suffix.lower()
    try:
        return _EXTENSIONS[suffix]
    except KeyError as exc:
        raise ReadError(f"cannot infer a supported format from extension {suffix!r}") from exc


def read(filename, file_format: str | None = None) -> Mesh:
    fmt = _format(filename, file_format)
    if fmt in ("obj", "off"):
        from ._text import read_obj, read_off

        return (read_obj if fmt == "obj" else read_off)(filename)
    if fmt == "ply":
        from ._ply import read_ply

        return read_ply(filename)
    if fmt == "stl":
        from ._stl import read_stl

        return read_stl(filename)
    if fmt == "vtk":
        from ._vtk import read_vtk

        return read_vtk(filename)
    if fmt in ("gltf", "glb"):
        from ._gltf import read_gltf

        return read_gltf(filename)
    raise ReadError(f"unsupported format {fmt!r}")


def write(
    filename,
    mesh: Mesh,
    file_format: str | None = None,
    **kwargs,
) -> None:
    fmt = _format(filename, file_format)
    for block in mesh.cells:
        if block.data.size and (
            block.data.min() < 0 or block.data.max() >= len(mesh.points)
        ):
            raise WriteError(f"{block.type} cell index is out of bounds")
    if fmt in ("obj", "off"):
        if kwargs:
            raise TypeError(f"unexpected writer option(s): {', '.join(sorted(kwargs))}")
        from ._text import write_obj, write_off

        (write_obj if fmt == "obj" else write_off)(filename, mesh)
        return
    if fmt == "ply":
        from ._ply import write_ply

        binary = kwargs.pop("binary", True)
        if kwargs:
            raise TypeError(f"unexpected writer option(s): {', '.join(sorted(kwargs))}")
        write_ply(filename, mesh, binary=binary)
        return
    if fmt == "stl":
        from ._stl import write_stl

        binary = kwargs.pop("binary", False)
        if kwargs:
            raise TypeError(f"unexpected writer option(s): {', '.join(sorted(kwargs))}")
        write_stl(filename, mesh, binary=binary)
        return
    if fmt == "vtk":
        from ._vtk import write_vtk

        binary = kwargs.pop("binary", True)
        if kwargs:
            raise TypeError(f"unexpected writer option(s): {', '.join(sorted(kwargs))}")
        write_vtk(filename, mesh, binary=binary)
        return
    if fmt in ("gltf", "glb"):
        if kwargs:
            raise TypeError(f"unexpected writer option(s): {', '.join(sorted(kwargs))}")
        from ._gltf import write_gltf

        write_gltf(filename, mesh, binary=(fmt == "glb"))
        return
    raise WriteError(f"unsupported format {fmt!r}")


def write_points_cells(
    filename,
    points,
    cells,
    point_data=None,
    cell_data=None,
    field_data=None,
    point_sets=None,
    cell_sets=None,
    file_format: str | None = None,
    **kwargs,
) -> None:
    mesh = Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data,
        field_data=field_data,
        point_sets=point_sets,
        cell_sets=cell_sets,
    )
    write(filename, mesh, file_format=file_format, **kwargs)
