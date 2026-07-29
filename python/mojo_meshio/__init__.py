from ._formats import ReadError, WriteError, read, write, write_points_cells
from ._geometry import gather_triangles, triangle_normals, weld_triangles
from ._mesh import CellBlock, Mesh

__all__ = [
    "CellBlock",
    "Mesh",
    "ReadError",
    "WriteError",
    "gather_triangles",
    "read",
    "triangle_normals",
    "weld_triangles",
    "write",
    "write_points_cells",
]

__version__ = "0.1.0"

extension_to_filetypes = {
    ".obj": ["obj"],
    ".off": ["off"],
    ".ply": ["ply"],
    ".stl": ["stl"],
    ".vtk": ["vtk"],
    ".gltf": ["gltf"],
    ".glb": ["glb"],
}
