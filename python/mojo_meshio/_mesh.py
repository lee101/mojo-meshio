from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CellBlock:
    type: str
    data: np.ndarray

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.data.ndim != 2:
            raise ValueError("cell connectivity must be a two-dimensional array")
        if self.data.dtype.kind not in "iu":
            raise TypeError("cell connectivity must contain integers")
        if self.data.size:
            limits = np.iinfo(np.int64)
            if self.data.min() < limits.min or self.data.max() > limits.max:
                raise OverflowError("cell connectivity cannot be represented as int64")

    @property
    def dim(self) -> int:
        return {
            "vertex": 0,
            "line": 1,
            "triangle": 2,
            "quad": 2,
            "tetra": 3,
            "hexahedron": 3,
            "wedge": 3,
            "pyramid": 3,
        }.get(self.type, 2)

    def __len__(self) -> int:
        return len(self.data)


class Mesh:
    def __init__(
        self,
        points,
        cells,
        point_data=None,
        cell_data=None,
        field_data=None,
        point_sets=None,
        cell_sets=None,
        gmsh_periodic=None,
        info=None,
    ) -> None:
        self.points = np.asarray(points)
        if self.points.ndim != 2 or self.points.shape[1] not in (2, 3):
            raise ValueError("points must have shape (n, 2) or (n, 3)")
        items = cells.items() if isinstance(cells, dict) else cells
        self.cells = [
            item if isinstance(item, CellBlock) else CellBlock(item[0], item[1])
            for item in items
        ]
        self.point_data = {
            key: np.asarray(value) for key, value in (point_data or {}).items()
        }
        self.cell_data = {
            key: [np.asarray(value) for value in values]
            for key, values in (cell_data or {}).items()
        }
        self.field_data = field_data or {}
        self.point_sets = point_sets or {}
        self.cell_sets = cell_sets or {}
        self.gmsh_periodic = gmsh_periodic
        self.info = info

    @property
    def cells_dict(self) -> dict[str, np.ndarray]:
        grouped: dict[str, list[np.ndarray]] = {}
        for block in self.cells:
            grouped.setdefault(block.type, []).append(block.data)
        return {
            key: values[0] if len(values) == 1 else np.concatenate(values)
            for key, values in grouped.items()
        }

    @property
    def cell_data_dict(self) -> dict[str, dict[str, np.ndarray]]:
        result: dict[str, dict[str, list[np.ndarray]]] = {}
        for name, arrays in self.cell_data.items():
            for block, array in zip(self.cells, arrays):
                result.setdefault(name, {}).setdefault(block.type, []).append(array)
        return {
            name: {
                cell_type: arrays[0] if len(arrays) == 1 else np.concatenate(arrays)
                for cell_type, arrays in typed.items()
            }
            for name, typed in result.items()
        }

    def write(self, path_or_buf, file_format: str | None = None, **kwargs) -> None:
        from ._formats import write

        write(path_or_buf, self, file_format=file_format, **kwargs)

    @classmethod
    def read(cls, path_or_buf, file_format: str | None = None):
        from ._formats import read

        return read(path_or_buf, file_format=file_format)

    def copy(self):
        return deepcopy(self)

    def get_cells_type(self, cell_type: str) -> np.ndarray:
        arrays = [block.data for block in self.cells if block.type == cell_type]
        if not arrays:
            return np.empty((0, 0), dtype=np.int64)
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)

    def get_cell_data(self, name: str, cell_type: str) -> np.ndarray:
        arrays = [
            array
            for block, array in zip(self.cells, self.cell_data[name])
            if block.type == cell_type
        ]
        if not arrays:
            raise ValueError(f"no cell data {name!r} for cell type {cell_type!r}")
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays)

    def __repr__(self) -> str:
        cell_lines = "\n".join(
            f"    {block.type}: {len(block.data)}" for block in self.cells
        )
        return (
            "<meshio mesh object>\n"
            f"  Number of points: {len(self.points)}\n"
            "  Number of cells:\n"
            f"{cell_lines}"
        )


def _path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(value)
