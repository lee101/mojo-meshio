from __future__ import annotations

import json

import meshio
import numpy as np
import pytest
from meshio.vtk import _vtk_42

import mojo_meshio as mm
from test_mesh import sample_mesh


def _sorted_rows(array):
    array = np.asarray(array)
    return np.asarray(sorted(map(tuple, array.tolist())))


def _assert_surface_geometry(actual, expected):
    assert np.asarray(actual.points) == pytest.approx(np.asarray(expected.points))
    for cell_type in ("triangle", "quad"):
        if cell_type in expected.cells_dict:
            assert np.array_equal(
                _sorted_rows(actual.cells_dict[cell_type]),
                _sorted_rows(expected.cells_dict[cell_type]),
            )


@pytest.mark.parametrize("extension", ["obj", "off"])
def test_text_writer_is_readable_by_upstream(extension, tmp_path):
    source = sample_mesh()
    if extension == "off":
        source = mm.Mesh(source.points, [source.cells[0]])
    path = tmp_path / f"mesh.{extension}"
    mm.write(path, source)
    _assert_surface_geometry(meshio.read(path), source)


@pytest.mark.parametrize("extension", ["obj", "off"])
def test_text_reader_reads_upstream_output(extension, tmp_path):
    source = sample_mesh()
    if extension == "off":
        source = mm.Mesh(source.points, [source.cells[0]])
    path = tmp_path / f"upstream.{extension}"
    meshio.write(path, meshio.Mesh(source.points, [(c.type, c.data) for c in source.cells]))
    _assert_surface_geometry(mm.read(path), source)


def test_off_reader_mixed_face_width_fallback(tmp_path):
    path = tmp_path / "mixed.off"
    path.write_text(
        "OFF\n5 2 0\n"
        "0 0 0\n1 0 0\n1 1 0\n0 1 0\n0.5 0.5 1\n"
        "3 0 1 4\n4 0 1 2 3\n"
    )
    got = mm.read(path)
    assert np.array_equal(got.cells_dict["triangle"], [[0, 1, 4]])
    assert np.array_equal(got.cells_dict["quad"], [[0, 1, 2, 3]])


@pytest.mark.parametrize("binary", [False, True])
def test_ply_bidirectional_upstream_parity(binary, tmp_path):
    source = sample_mesh()
    ours = tmp_path / f"ours-{binary}.ply"
    mm.write(ours, source, binary=binary)
    upstream_read = meshio.read(ours)
    _assert_surface_geometry(upstream_read, source)
    assert upstream_read.point_data["temperature"] == pytest.approx(
        source.point_data["temperature"]
    )

    upstream = tmp_path / f"upstream-{binary}.ply"
    meshio.write(
        upstream,
        meshio.Mesh(
            source.points,
            [(c.type, c.data) for c in source.cells],
            point_data=source.point_data,
        ),
        binary=binary,
    )
    ours_read = mm.read(upstream)
    _assert_surface_geometry(ours_read, source)
    assert ours_read.point_data["temperature"] == pytest.approx(
        source.point_data["temperature"]
    )


@pytest.mark.parametrize("binary", [False, True])
def test_stl_bidirectional_geometry_parity(binary, tmp_path):
    source = sample_mesh()
    triangles = mm.Mesh(source.points, [source.cells[0]])
    ours = tmp_path / f"ours-{binary}.stl"
    mm.write(ours, triangles, binary=binary)
    upstream_read = meshio.read(ours)
    assert len(upstream_read.cells_dict["triangle"]) == 2
    assert {
        tuple(row)
        for row in upstream_read.points[upstream_read.cells_dict["triangle"]].reshape(-1, 3)
    } == {tuple(row) for row in source.points[[0, 1, 2, 3]]}

    upstream = tmp_path / f"upstream-{binary}.stl"
    meshio.write(
        upstream,
        meshio.Mesh(source.points, [("triangle", source.cells[0].data)]),
        binary=binary,
    )
    ours_read = mm.read(upstream)
    assert len(ours_read.cells_dict["triangle"]) == 2
    assert ours_read.cell_data["facet_normals"][0] == pytest.approx(
        np.array([[0, 0, 1], [0, 0, 1]])
    )


@pytest.mark.parametrize("binary", [False, True])
def test_vtk_writer_is_readable_by_upstream_with_data(binary, tmp_path):
    source = sample_mesh()
    path = tmp_path / f"ours-{binary}.vtk"
    mm.write(path, source, binary=binary)
    got = meshio.read(path)
    _assert_surface_geometry(got, source)
    assert got.point_data["temperature"] == pytest.approx(source.point_data["temperature"])
    assert got.cell_data_dict["region"]["triangle"] == pytest.approx([10, 11])
    assert got.cell_data_dict["region"]["quad"] == pytest.approx([12])


@pytest.mark.parametrize("binary", [False, True])
def test_vtk_reader_reads_upstream_v51_with_data(binary, tmp_path):
    source = sample_mesh()
    path = tmp_path / f"upstream-{binary}.vtk"
    meshio.write(
        path,
        meshio.Mesh(
            source.points,
            [(c.type, c.data) for c in source.cells],
            point_data=source.point_data,
            cell_data=source.cell_data,
        ),
        binary=binary,
    )
    got = mm.read(path)
    _assert_surface_geometry(got, source)
    assert got.point_data["temperature"] == pytest.approx(source.point_data["temperature"])
    assert got.cell_data_dict["region"]["triangle"] == pytest.approx([10, 11])
    assert got.cell_data_dict["region"]["quad"] == pytest.approx([12])


@pytest.mark.parametrize("binary", [False, True])
def test_vtk_reader_reads_upstream_v42(binary, tmp_path):
    source = sample_mesh()
    path = tmp_path / f"upstream-v42-{binary}.vtk"
    _vtk_42.write(
        path,
        meshio.Mesh(source.points, [(c.type, c.data) for c in source.cells]),
        binary=binary,
    )
    _assert_surface_geometry(mm.read(path), source)


@pytest.mark.parametrize(
    ("cell_type", "width"),
    [
        ("vertex", 1),
        ("line", 2),
        ("triangle", 3),
        ("quad", 4),
        ("tetra", 4),
        ("pyramid", 5),
        ("wedge", 6),
        ("hexahedron", 8),
    ],
)
def test_vtk_supported_linear_cell_types(cell_type, width, tmp_path):
    points = np.column_stack(
        (np.arange(width, dtype=np.float64), np.zeros(width), np.zeros(width))
    )
    cells = np.arange(width, dtype=np.int64).reshape(1, width)
    path = tmp_path / f"{cell_type}.vtk"
    mm.write(path, mm.Mesh(points, [(cell_type, cells)]), binary=True)
    got = mm.read(path)
    assert np.array_equal(got.cells_dict[cell_type], cells)
    upstream = meshio.read(path)
    assert np.array_equal(upstream.cells_dict[cell_type], cells)


@pytest.mark.parametrize("extension", ["gltf", "glb"])
def test_gltf_roundtrip_and_schema(extension, tmp_path):
    source = sample_mesh()
    triangles = mm.Mesh(source.points, [source.cells[0]])
    path = tmp_path / f"mesh.{extension}"
    mm.write(path, triangles)
    got = mm.read(path)
    assert got.points == pytest.approx(source.points.astype(np.float32))
    assert np.array_equal(got.cells_dict["triangle"], source.cells[0].data)
    if extension == "gltf":
        document = json.loads(path.read_text())
        assert document["asset"]["version"] == "2.0"
        assert document["meshes"][0]["primitives"][0]["mode"] == 4
        assert document["accessors"][0]["type"] == "VEC3"


def test_obj_negative_indices(tmp_path):
    path = tmp_path / "negative.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n")
    got = mm.read(path)
    assert np.array_equal(got.cells_dict["triangle"], [[0, 1, 2]])


def test_obj_slash_indices_attributes_and_primitive_types(tmp_path):
    path = tmp_path / "features.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 0.5 1\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvt 0.5 0.5\n"
        "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
        "p 1\nl 1 2 3\nf 1/1/1 2/2/2 3/3/3 4/4/4\n"
        "f 1 2 3 4 5\n"
    )
    got = mm.read(path)
    assert np.array_equal(got.cells_dict["vertex"], [[0]])
    assert np.array_equal(got.cells_dict["line"], [[0, 1], [1, 2]])
    assert np.array_equal(got.cells_dict["quad"], [[0, 1, 2, 3]])
    assert np.array_equal(got.cells_dict["polygon5"], [[0, 1, 2, 3, 4]])
    assert got.point_data["obj:vt"].shape == (5, 2)
    assert got.point_data["obj:vn"].shape == (5, 3)


def test_ply_binary_big_endian_input(tmp_path):
    path = tmp_path / "big-endian.ply"
    header = (
        b"ply\nformat binary_big_endian 1.0\n"
        b"element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
        b"property float temperature\n"
        b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    )
    vertices = np.array(
        [(0, 0, 0, 10), (1, 0, 0, 20), (0, 1, 0, 30)],
        dtype=[("x", ">f4"), ("y", ">f4"), ("z", ">f4"), ("t", ">f4")],
    )
    path.write_bytes(header + vertices.tobytes() + b"\x03" + np.array([0, 1, 2], dtype=">i4").tobytes())
    got = mm.read(path)
    assert np.array_equal(got.cells_dict["triangle"], [[0, 1, 2]])
    assert np.array_equal(got.point_data["temperature"], [10, 20, 30])


@pytest.mark.parametrize(
    ("cell_type", "cells"),
    [
        ("vertex", [[0], [1]]),
        ("line", [[0, 1], [1, 2]]),
        ("triangle", [[0, 1, 2]]),
    ],
)
@pytest.mark.parametrize("extension", ["gltf", "glb"])
def test_gltf_supported_primitive_modes(cell_type, cells, extension, tmp_path):
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    path = tmp_path / f"primitive.{extension}"
    mm.write(path, mm.Mesh(points, [(cell_type, np.asarray(cells, dtype=np.int64))]))
    got = mm.read(path)
    assert np.array_equal(got.cells_dict[cell_type], cells)


def test_stl_binary_detection_does_not_trust_solid_prefix(tmp_path):
    source = sample_mesh()
    path = tmp_path / "solid-header.stl"
    mm.write(path, mm.Mesh(source.points, [source.cells[0]]), binary=True)
    raw = path.read_bytes()
    path.write_bytes(b"solid" + raw[5:])
    assert len(mm.read(path).cells_dict["triangle"]) == 2
