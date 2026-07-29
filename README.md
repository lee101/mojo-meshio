# mojo-meshio

`mojo-meshio` is a standalone mesh format library with meshio-shaped Python APIs and
compiled Mojo geometry kernels. It reads and writes OBJ, OFF, PLY, STL, legacy VTK,
and GLTF 2.0 while keeping point and connectivity arrays as NumPy buffers.

The package is intended for geometry pipelines that need a compact format layer without
pulling in upstream meshio at runtime. Upstream `meshio` 5.3.5 is installed only for the
test suite and benchmarks.

## Covered subset

- `Mesh`, `CellBlock`, `read`, `write`, and `write_points_cells`, with the same core
  signatures and names as meshio.
- Zero-copy `Mesh` construction: compatible NumPy point and cell arrays are retained,
  not converted or copied.
- OBJ: vertices, per-vertex texture coordinates and normals, points, lines, triangles,
  quads, polygons, negative indices, and slash-form face indices.
- OFF: ASCII polygonal surface meshes.
- PLY: ASCII and binary little-/big-endian input, ASCII and little-endian output,
  polygonal faces, and scalar point properties.
- STL: ASCII and binary triangle meshes, facet normals, exact stable vertex welding,
  and robust binary detection.
- Legacy VTK unstructured grids: VTK 4.2 and 5.1 input, ASCII and binary 5.1 output,
  vertex, line, triangle, quad, tetrahedron, pyramid, wedge, and hexahedron cells,
  and scalar numeric `FIELD` point/cell data.
- GLTF 2.0: embedded-buffer `.gltf` and binary `.glb`, indexed POSITION primitives,
  and point, line, and triangle modes.
- Public Mojo-backed `triangle_normals`, `gather_triangles`, and `weld_triangles`
  operations.

OBJ, OFF, PLY, STL, and VTK writes are read by upstream meshio in the test suite, and
upstream-generated files exercise the corresponding readers. Meshio 5.3.5 has no GLTF
reader or writer, so GLTF point, line, and triangle primitives are checked through
independent `.gltf` and `.glb` round trips and container/accessor assertions.

This is not the complete meshio format matrix. It does not cover XML VTU, XDMF, Gmsh,
Exodus, or other formats outside the list above. It also omits OBJ materials, general PLY
elements/list-valued metadata, STL colors, VTK structured datasets and higher-order
fields, and GLTF materials, skins, animation, sparse accessors, and attributes other than
POSITION. Format-specific metadata that cannot be represented by the covered subset is
not preserved.

## Install and build

From the repository root:

```bash
pixi install
pixi run build
pixi run test
```

The build task compiles the single Mojo compilation unit to
`dist/libmojo-meshio.so`. Pixi activates the repository's `python/` directory on
`PYTHONPATH`; run the examples below from the repository root with `pixi run python`.

## Usage

```python
import numpy as np
import mojo_meshio as meshio

points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
])
cells = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)

mesh = meshio.Mesh(points, [("triangle", cells)])
mesh.write("square.ply", binary=True)
roundtrip = meshio.read("square.ply")

assert np.allclose(roundtrip.points, points)
assert np.array_equal(roundtrip.cells_dict["triangle"], cells)
```

Use the Mojo geometry operations directly when no file I/O is needed:

```python
normals = meshio.triangle_normals(points, cells)
triangles = meshio.gather_triangles(points, cells)
unique_points, remapped_cells = meshio.weld_triangles(triangles)
```

## How it works

Python handles format headers, sections, and metadata. Mojo scans dense ASCII number
streams, computes and gathers triangle geometry, and welds STL vertices with an
open-addressed exact-bit hash table. STL welding is stable: the first occurrence of each
coordinate triple determines its output index.

The shared library uses a small C ABI through `ctypes`. Buffers cross the ABI as integer
addresses and are reconstructed as `UnsafePointer[..., AnyOrigin[mut=True]]` inside the
exported functions. Python/NumPy owns every allocation; Mojo neither allocates nor frees
those buffers. Geometry calls are zero-copy when inputs are C-contiguous `float64` points
and `int64` connectivity. Other supported dtypes or layouts are normalized at the
wrapper boundary. GLTF accessors are returned as direct `np.frombuffer` views when they
are tightly packed.

## Benchmarks

Measured on 2026-07-29 with an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64.
Times are the best of five warm runs from `pixi run bench` on this machine.
Format reads use a warm filesystem page cache, equally for both implementations.

| benchmark | mojo-meshio | reference | relative |
| --- | ---: | ---: | ---: |
| triangle normals (500k) | 12.29 ms | 160.10 ms (NumPy) | 13.03x faster |
| indexed triangle gather (500k) | 6.94 ms | 51.99 ms (NumPy) | 7.49x faster |
| binary STL read (150k triangles) | 60.13 ms | 777.97 ms (meshio) | 12.94x faster |
| ASCII OFF read (150k triangles) | 102.20 ms | 1728.94 ms (meshio) | 16.92x faster |

The NumPy geometry references materialize advanced-indexing intermediates; the Mojo
kernels stream through the indexed arrays once. The format results include parsing,
allocation, and geometry reconstruction, not only kernel time.

No GPU path is provided.

Run the benchmark on the current machine with:

```bash
pixi run bench
```

MIT licensed.
