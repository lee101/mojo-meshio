from __future__ import annotations

import numpy as np
import pytest

import mojo_meshio as mm
from mojo_meshio._lib import scan_f64, scan_i64


def sample_mesh() -> mm.Mesh:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mm.Mesh(
        points,
        [
            ("triangle", np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)),
            ("quad", np.array([[0, 1, 2, 3]], dtype=np.int64)),
        ],
        point_data={"temperature": np.arange(5, dtype=np.float64)},
        cell_data={
            "region": [
                np.array([10, 11], dtype=np.int64),
                np.array([12], dtype=np.int64),
            ]
        },
    )


def test_mesh_keeps_compatible_numpy_buffers_zero_copy():
    points = np.zeros((4, 3), dtype=np.float64)
    cells = np.array([[0, 1, 2]], dtype=np.int64)
    mesh = mm.Mesh(points, [("triangle", cells)])
    assert mesh.points is points
    assert mesh.cells[0].data is cells
    points[0, 0] = 9.0
    cells[0, 0] = 3
    assert mesh.points[0, 0] == 9.0
    assert mesh.cells[0].data[0, 0] == 3


def test_cells_dict_concatenates_repeated_blocks():
    mesh = mm.Mesh(
        np.zeros((4, 3)),
        [
            ("triangle", np.array([[0, 1, 2]])),
            ("triangle", np.array([[0, 2, 3]])),
        ],
    )
    assert np.array_equal(mesh.cells_dict["triangle"], [[0, 1, 2], [0, 2, 3]])


def test_cell_blocks_reject_silent_connectivity_narrowing():
    with pytest.raises(TypeError, match="integers"):
        mm.CellBlock("triangle", [[0.0, 1.0, 2.0]])
    with pytest.raises(OverflowError, match="int64"):
        mm.CellBlock("triangle", np.array([[0, 1, 2**63]], dtype=np.uint64))


def test_triangle_normals_match_numpy_cross_product():
    mesh = sample_mesh()
    cells = mesh.cells[0].data
    got = mm.triangle_normals(mesh.points, cells)
    triangles = mesh.points[cells]
    expected = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    assert got == pytest.approx(expected)


def test_triangle_normals_define_degenerate_result_as_zero():
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    assert np.array_equal(mm.triangle_normals(points, [[0, 1, 2]]), [[0, 0, 0]])


@pytest.mark.parametrize("operation", [mm.triangle_normals, mm.gather_triangles])
@pytest.mark.parametrize("cells", [[[-1, 1, 2]], [[0, 1, 3]]])
def test_geometry_rejects_out_of_bounds_indices(operation, cells):
    with pytest.raises(IndexError, match="out of bounds"):
        operation(np.zeros((3, 3)), cells)


@pytest.mark.parametrize("operation", [mm.triangle_normals, mm.gather_triangles])
def test_geometry_rejects_silent_connectivity_narrowing(operation):
    points = np.zeros((3, 3))
    with pytest.raises(TypeError, match="integers"):
        operation(points, [[0.0, 1.0, 2.0]])
    with pytest.raises(OverflowError, match="int64"):
        operation(points, np.array([[0, 1, 2**63]], dtype=np.uint64))


def test_geometry_empty_inputs_are_safe():
    points = np.empty((0, 3))
    cells = np.empty((0, 3), dtype=np.int64)
    assert mm.triangle_normals(points, cells).shape == (0, 3)
    assert mm.gather_triangles(points, cells).shape == (0, 3, 3)
    welded = mm.weld_triangles(np.empty((0, 3, 3)))
    assert welded[0].shape == (0, 3)
    assert welded[1].shape == (0, 3)


def test_weld_triangles_is_stable_and_exact():
    vertices = np.array(
        [
            [[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0]],
            [[0.0, 0, 0], [1.0, 1, 0], [0.0, 1, 0]],
        ]
    )
    points, cells = mm.weld_triangles(vertices)
    assert np.array_equal(points, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    assert np.array_equal(cells, [[0, 1, 2], [0, 2, 3]])
    assert np.array_equal(mm.gather_triangles(points, cells), vertices)


def test_weld_triangles_treats_signed_zero_as_the_same_coordinate():
    vertices = np.array(
        [
            [[-0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]],
            [[+0.0, 0, 0], [0.0, 1, 0], [1.0, 1, 0]],
        ]
    )
    points, cells = mm.weld_triangles(vertices)
    assert len(points) == 4
    assert cells[0, 0] == cells[1, 0]


@pytest.mark.parametrize("triangle_count", [1, 2, 16383, 16384])
def test_gather_triangles_simd_tail_and_parallel_threshold(triangle_count):
    cells = np.arange(triangle_count * 3, dtype=np.int64).reshape(-1, 3)
    points = np.arange(triangle_count * 9, dtype=np.float64).reshape(-1, 3)
    assert np.array_equal(mm.gather_triangles(points, cells), points[cells])


def test_gather_triangles_dispatches_only_at_parallel_threshold(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import mojo_meshio._geometry as geometry

    points = np.arange(45, dtype=np.float64).reshape(-1, 3)
    cells = np.arange(15, dtype=np.int64).reshape(-1, 3)

    monkeypatch.setattr(geometry, "_GATHER_PARALLEL_THRESHOLD", 5)
    monkeypatch.setattr(geometry, "_GATHER_WORKERS", 2)
    monkeypatch.setattr(
        geometry,
        "_executor",
        lambda: (_ for _ in ()).throw(AssertionError("parallel dispatch below threshold")),
    )
    assert np.array_equal(mm.gather_triangles(points, cells[:4]), points[cells[:4]])

    dispatches = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        monkeypatch.setattr(
            geometry,
            "_executor",
            lambda: dispatches.append(True) or executor,
        )
        assert np.array_equal(mm.gather_triangles(points, cells), points[cells])
    assert dispatches


def test_mesh_write_method(tmp_path):
    path = tmp_path / "mesh.off"
    sample_mesh().write(path)
    assert len(mm.read(path).points) == 5


def test_ascii_numeric_kernels_handle_signs_exponents_and_comments():
    assert scan_f64(b"1 -2.5 +3e2 4E-2 # ignored\n5", 5) == pytest.approx(
        [1.0, -2.5, 300.0, 0.04, 5.0]
    )
    assert np.array_equal(scan_i64(b"-4 +5 6 # ignored\n7", 4), [-4, 5, 6, 7])
    assert scan_f64(b"", 0).shape == (0,)
    assert scan_i64(b"", 0).shape == (0,)
    assert np.array_equal(
        scan_i64(b"-9223372036854775808 9223372036854775807", 2),
        [np.iinfo(np.int64).min, np.iinfo(np.int64).max],
    )
    with pytest.raises(ValueError, match="invalid integer"):
        scan_i64(b"9223372036854775808", 1)
    with pytest.raises(ValueError, match="invalid numeric"):
        scan_f64(b"1e999999999999999999999", 1)


def test_writer_rejects_unknown_options(tmp_path):
    with pytest.raises(TypeError, match="unexpected writer option"):
        mm.write(tmp_path / "mesh.off", sample_mesh(), ignored=True)


def test_writer_rejects_out_of_bounds_connectivity(tmp_path):
    mesh = mm.Mesh(np.zeros((3, 3)), [("triangle", [[0, 1, 3]])])
    with pytest.raises(mm.WriteError, match="out of bounds"):
        mm.write(tmp_path / "invalid.obj", mesh)


def test_meshio_compatible_helpers(tmp_path):
    source = sample_mesh()
    assert np.array_equal(source.get_cells_type("triangle"), source.cells[0].data)
    assert np.array_equal(source.get_cell_data("region", "quad"), [12])
    path = tmp_path / "points-cells.obj"
    mm.write_points_cells(path, source.points, [source.cells[0]])
    assert len(mm.Mesh.read(path).cells_dict["triangle"]) == 2
