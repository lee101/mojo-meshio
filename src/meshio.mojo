"""Numeric mesh kernels exposed through a C ABI."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def is_space(c: UInt8) -> Bool:
    return c == UInt8(32) or c == UInt8(9) or c == UInt8(10) or c == UInt8(13)


def skip_space_and_comments(src: BPtr, n: Int, start: Int) -> Int:
    var i = start
    while i < n:
        if is_space(src[i]):
            i += 1
            continue
        if src[i] == UInt8(35):
            while i < n and src[i] != UInt8(10):
                i += 1
            continue
        break
    return i


def parse_f64(src: BPtr, n: Int, dst: FPtr, capacity: Int) -> Int:
    var i = 0
    var count = 0
    while True:
        i = skip_space_and_comments(src, n, i)
        if i >= n:
            return count
        if count >= capacity:
            return -1

        var negative = False
        if src[i] == UInt8(43) or src[i] == UInt8(45):
            negative = src[i] == UInt8(45)
            i += 1

        var value = Float64(0.0)
        var digits = 0
        while i < n and src[i] >= UInt8(48) and src[i] <= UInt8(57):
            value = value * 10.0 + Float64(Int(src[i]) - 48)
            digits += 1
            i += 1

        if i < n and src[i] == UInt8(46):
            i += 1
            var place = Float64(0.1)
            while i < n and src[i] >= UInt8(48) and src[i] <= UInt8(57):
                value += Float64(Int(src[i]) - 48) * place
                place *= 0.1
                digits += 1
                i += 1

        if digits == 0:
            return -(i + 2)

        var exponent = 0
        var exponent_negative = False
        if i < n and (src[i] == UInt8(101) or src[i] == UInt8(69)):
            i += 1
            if i < n and (src[i] == UInt8(43) or src[i] == UInt8(45)):
                exponent_negative = src[i] == UInt8(45)
                i += 1
            var exponent_digits = 0
            while i < n and src[i] >= UInt8(48) and src[i] <= UInt8(57):
                if exponent > 10000:
                    return -(i + 2)
                exponent = exponent * 10 + Int(src[i]) - 48
                exponent_digits += 1
                i += 1
            if exponent_digits == 0:
                return -(i + 2)

        var scale = Float64(1.0)
        while exponent >= 8:
            scale *= 100000000.0
            exponent -= 8
        while exponent > 0:
            scale *= 10.0
            exponent -= 1
        if exponent_negative:
            value /= scale
        else:
            value *= scale
        if negative:
            value = -value
        dst[count] = value
        count += 1
        if i < n and not is_space(src[i]) and src[i] != UInt8(35):
            return -(i + 2)


def parse_i64(src: BPtr, n: Int, dst: IPtr, capacity: Int) -> Int:
    var i = 0
    var count = 0
    while True:
        i = skip_space_and_comments(src, n, i)
        if i >= n:
            return count
        if count >= capacity:
            return -1
        var negative = False
        if src[i] == UInt8(43) or src[i] == UInt8(45):
            negative = src[i] == UInt8(45)
            i += 1
        var value = Int64(0)
        var digits = 0
        while i < n and src[i] >= UInt8(48) and src[i] <= UInt8(57):
            var digit = Int64(Int(src[i]) - 48)
            var last_digit = Int64(8) if negative else Int64(7)
            if (
                value > Int64(922337203685477580)
                or (
                    value == Int64(922337203685477580)
                    and digit > last_digit
                )
            ):
                return -(i + 2)
            value = value * 10 + digit
            digits += 1
            i += 1
        if digits == 0:
            return -(i + 2)
        dst[count] = -value if negative else value
        count += 1
        if i < n and not is_space(src[i]) and src[i] != UInt8(35):
            return -(i + 2)


def triangle_normals(points: FPtr, cells: IPtr, normals: FPtr, count: Int) -> Int:
    var degenerate = 0
    for triangle in range(count):
        var ia = Int(cells[triangle * 3]) * 3
        var ib = Int(cells[triangle * 3 + 1]) * 3
        var ic = Int(cells[triangle * 3 + 2]) * 3
        var ax = points[ib] - points[ia]
        var ay = points[ib + 1] - points[ia + 1]
        var az = points[ib + 2] - points[ia + 2]
        var bx = points[ic] - points[ia]
        var by = points[ic + 1] - points[ia + 1]
        var bz = points[ic + 2] - points[ia + 2]
        var nx = ay * bz - az * by
        var ny = az * bx - ax * bz
        var nz = ax * by - ay * bx
        var length = sqrt(nx * nx + ny * ny + nz * nz)
        if length == 0.0:
            degenerate += 1
            normals[triangle * 3] = 0.0
            normals[triangle * 3 + 1] = 0.0
            normals[triangle * 3 + 2] = 0.0
        else:
            normals[triangle * 3] = nx / length
            normals[triangle * 3 + 1] = ny / length
            normals[triangle * 3 + 2] = nz / length
    return degenerate


def mix64(value: UInt64) -> UInt64:
    var x = value
    x = (x ^ (x >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> 27)) * UInt64(0x94D049BB133111EB)
    return x ^ (x >> 31)


def weld_triangles(
    vertices: FPtr,
    triangle_count: Int,
    points: FPtr,
    cells: IPtr,
    table: IPtr,
    table_capacity: Int,
) -> Int:
    var unique_count = 0
    for vertex in range(triangle_count * 3):
        var source = vertex * 3
        var x_bits = vertices[source].to_bits[DType.uint64]()
        var y_bits = vertices[source + 1].to_bits[DType.uint64]()
        var z_bits = vertices[source + 2].to_bits[DType.uint64]()
        if vertices[source] == 0.0:
            x_bits = UInt64(0)
        if vertices[source + 1] == 0.0:
            y_bits = UInt64(0)
        if vertices[source + 2] == 0.0:
            z_bits = UInt64(0)
        var hash_value = mix64(x_bits) ^ mix64(y_bits + UInt64(0x9E3779B97F4A7C15))
        hash_value ^= mix64(z_bits + UInt64(0xD1B54A32D192ED03))
        var slot = Int(hash_value % UInt64(table_capacity))
        while True:
            var entry = Int(table[slot])
            if entry < 0:
                var target = unique_count * 3
                points[target] = vertices[source]
                points[target + 1] = vertices[source + 1]
                points[target + 2] = vertices[source + 2]
                table[slot] = Int64(unique_count)
                cells[vertex] = Int64(unique_count)
                unique_count += 1
                break
            var target = entry * 3
            if (
                points[target] == vertices[source]
                and points[target + 1] == vertices[source + 1]
                and points[target + 2] == vertices[source + 2]
            ):
                cells[vertex] = Int64(entry)
                break
            slot += 1
            if slot == table_capacity:
                slot = 0
    return unique_count


def gather_triangle_vertices(
    points: FPtr, cells: IPtr, vertices: FPtr, start: Int, stop: Int
):
    comptime W = simd_width_of[DType.float64]()
    var vertex = start
    var vector_stop = stop - (stop - start) % W
    while vertex < vector_stop:
        var offsets = cells.load[width=W](vertex) * Int64(3)
        var x = points.gather(offsets)
        var y = points.gather(offsets + Int64(1))
        var z = points.gather(offsets + Int64(2))
        (vertices + vertex * 3).strided_store[width=W](x, 3)
        (vertices + vertex * 3 + 1).strided_store[width=W](y, 3)
        (vertices + vertex * 3 + 2).strided_store[width=W](z, 3)
        vertex += W
    while vertex < stop:
        var source = Int(cells[vertex]) * 3
        var target = vertex * 3
        vertices[target] = points[source]
        vertices[target + 1] = points[source + 1]
        vertices[target + 2] = points[source + 2]
        vertex += 1


def gather_triangles(points: FPtr, cells: IPtr, vertices: FPtr, count: Int):
    comptime PARALLEL_THRESHOLD = 16384
    comptime CHUNK_TRIANGLES = 4096
    var vertex_count = count * 3
    if count < PARALLEL_THRESHOLD:
        gather_triangle_vertices(points, cells, vertices, 0, vertex_count)
        return

    var chunk_count = (count + CHUNK_TRIANGLES - 1) // CHUNK_TRIANGLES

    for chunk in range(chunk_count):
        var triangle_start = chunk * CHUNK_TRIANGLES
        var triangle_stop = min(triangle_start + CHUNK_TRIANGLES, count)
        gather_triangle_vertices(
            points,
            cells,
            vertices,
            triangle_start * 3,
            triangle_stop * 3,
        )


@export("mmi_parse_f64")
def mmi_parse_f64(src: Int, n: Int, dst: Int, capacity: Int) abi("C") -> Int:
    return parse_f64(
        BPtr(unsafe_from_address=src), n,
        FPtr(unsafe_from_address=dst), capacity,
    )


@export("mmi_parse_i64")
def mmi_parse_i64(src: Int, n: Int, dst: Int, capacity: Int) abi("C") -> Int:
    return parse_i64(
        BPtr(unsafe_from_address=src), n,
        IPtr(unsafe_from_address=dst), capacity,
    )


@export("mmi_triangle_normals")
def mmi_triangle_normals(
    points: Int, cells: Int, normals: Int, count: Int
) abi("C") -> Int:
    return triangle_normals(
        FPtr(unsafe_from_address=points),
        IPtr(unsafe_from_address=cells),
        FPtr(unsafe_from_address=normals),
        count,
    )


@export("mmi_weld_triangles")
def mmi_weld_triangles(
    vertices: Int,
    triangle_count: Int,
    points: Int,
    cells: Int,
    table: Int,
    table_capacity: Int,
) abi("C") -> Int:
    return weld_triangles(
        FPtr(unsafe_from_address=vertices),
        triangle_count,
        FPtr(unsafe_from_address=points),
        IPtr(unsafe_from_address=cells),
        IPtr(unsafe_from_address=table),
        table_capacity,
    )


@export("mmi_gather_triangles")
def mmi_gather_triangles(
    points: Int, cells: Int, vertices: Int, count: Int
) abi("C"):
    gather_triangles(
        FPtr(unsafe_from_address=points),
        IPtr(unsafe_from_address=cells),
        FPtr(unsafe_from_address=vertices),
        count,
    )
