"""Compute kernels for flat, IVF, and product-quantized vector search."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime W = 8
comptime PW = simd_width_of[DType.float64]()
comptime FPtr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def iptr(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def l2(a: FPtr, b: FPtr, d: Int) -> Float32:
    var vacc = SIMD[DType.float32, W](0.0)
    var j = 0
    while j + W <= d:
        var delta = a.load[width=W](j) - b.load[width=W](j)
        vacc += delta * delta
        j += W
    var acc = vacc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        acc += delta * delta
        j += 1
    return acc


@always_inline
def pq_l2(a: FPtr, b: FPtr, d: Int) -> Float32:
    var vacc = SIMD[DType.float32, PW](0.0)
    var j = 0
    while j + PW <= d:
        var delta = a.load[width=PW](j) - b.load[width=PW](j)
        vacc += delta * delta
        j += PW
    var acc = vacc.reduce_add()
    while j < d:
        var delta = a[j] - b[j]
        acc += delta * delta
        j += 1
    return acc


def inner_product(a: FPtr, b: FPtr, d: Int) -> Float32:
    var vacc = SIMD[DType.float32, W](0.0)
    var j = 0
    while j + W <= d:
        vacc += a.load[width=W](j) * b.load[width=W](j)
        j += W
    var acc = vacc.reduce_add()
    while j < d:
        acc += a[j] * b[j]
        j += 1
    return acc


def score(a: FPtr, b: FPtr, d: Int, metric: Int) -> Float32:
    if metric == 0:
        return inner_product(a, b, d)
    return l2(a, b, d)


def is_better(value: Float32, ident: Int64, old: Float32, old_id: Int64, metric: Int) -> Bool:
    if metric == 0:
        return value > old or (value == old and (old_id < 0 or ident < old_id))
    return value < old or (value == old and (old_id < 0 or ident < old_id))


def init_topk(distances: FPtr, ids: IPtr, base: Int, k: Int, metric: Int):
    var sentinel = Float32(3.4028234663852886e38)
    if metric == 0:
        sentinel = -sentinel
    for s in range(k):
        distances[base + s] = sentinel
        ids[base + s] = -1


def insert_topk(
    distances: FPtr,
    ids: IPtr,
    base: Int,
    k: Int,
    value: Float32,
    ident: Int64,
    metric: Int,
):
    if not is_better(value, ident, distances[base + k - 1], ids[base + k - 1], metric):
        return
    var s = k - 1
    while s > 0 and is_better(
        value, ident, distances[base + s - 1], ids[base + s - 1], metric
    ):
        distances[base + s] = distances[base + s - 1]
        ids[base + s] = ids[base + s - 1]
        s -= 1
    distances[base + s] = value
    ids[base + s] = ident


@export("mf_flat_search")
def mf_flat_search(
    xb_addr: Int,
    xq_addr: Int,
    dist_addr: Int,
    ids_addr: Int,
    nb: Int,
    nq: Int,
    d: Int,
    k: Int,
    metric: Int,
) abi("C"):
    var xb = fp(xb_addr)
    var xq = fp(xq_addr)
    var distances = fp(dist_addr)
    var ids = iptr(ids_addr)
    for q in range(nq):
        var base = q * k
        init_topk(distances, ids, base, k, metric)
        for r in range(nb):
            var value = score(xq + q * d, xb + r * d, d, metric)
            insert_topk(distances, ids, base, k, value, Int64(r), metric)


@export("mf_range_count")
def mf_range_count(
    xb_addr: Int,
    xq_addr: Int,
    counts_addr: Int,
    nb: Int,
    nq: Int,
    d: Int,
    radius: Float32,
    metric: Int,
) abi("C"):
    var xb = fp(xb_addr)
    var xq = fp(xq_addr)
    var counts = iptr(counts_addr)
    for q in range(nq):
        var count = Int64(0)
        for r in range(nb):
            var value = score(xq + q * d, xb + r * d, d, metric)
            if (metric == 0 and value > radius) or (metric != 0 and value < radius):
                count += 1
        counts[q] = count


@export("mf_range_fill")
def mf_range_fill(
    xb_addr: Int,
    xq_addr: Int,
    lims_addr: Int,
    dist_addr: Int,
    ids_addr: Int,
    nb: Int,
    nq: Int,
    d: Int,
    radius: Float32,
    metric: Int,
) abi("C"):
    var xb = fp(xb_addr)
    var xq = fp(xq_addr)
    var lims = iptr(lims_addr)
    var distances = fp(dist_addr)
    var ids = iptr(ids_addr)
    for q in range(nq):
        var pos = Int(lims[q])
        for r in range(nb):
            var value = score(xq + q * d, xb + r * d, d, metric)
            if (metric == 0 and value > radius) or (metric != 0 and value < radius):
                distances[pos] = value
                ids[pos] = Int64(r)
                pos += 1


@export("mf_assign")
def mf_assign(
    x_addr: Int,
    centers_addr: Int,
    labels_addr: Int,
    n: Int,
    d: Int,
    k: Int,
    metric: Int,
) abi("C"):
    var x = fp(x_addr)
    var centers = fp(centers_addr)
    var labels = iptr(labels_addr)
    for r in range(n):
        var best = score(x + r * d, centers, d, metric)
        var best_id = Int64(0)
        for c in range(1, k):
            var value = score(x + r * d, centers + c * d, d, metric)
            if is_better(value, Int64(c), best, best_id, metric):
                best = value
                best_id = Int64(c)
        labels[r] = best_id


@export("mf_kmeans")
def mf_kmeans(
    x_addr: Int,
    centers_addr: Int,
    labels_addr: Int,
    sums_addr: Int,
    counts_addr: Int,
    n: Int,
    d: Int,
    k: Int,
    max_iter: Int,
    tol: Float32,
    metric: Int,
) abi("C") -> Float32:
    var x = fp(x_addr)
    var centers = fp(centers_addr)
    var labels = iptr(labels_addr)
    var sums = fp(sums_addr)
    var counts = iptr(counts_addr)
    var objective = Float32(0.0)
    for _ in range(max_iter):
        for i in range(k * d):
            sums[i] = 0.0
        for c in range(k):
            counts[c] = 0
        objective = 0.0
        for r in range(n):
            var best = score(x + r * d, centers, d, metric)
            var best_id = Int64(0)
            for c in range(1, k):
                var value = score(x + r * d, centers + c * d, d, metric)
                if is_better(value, Int64(c), best, best_id, metric):
                    best = value
                    best_id = Int64(c)
            labels[r] = best_id
            if metric == 0:
                objective += best
            else:
                objective += best
            counts[Int(best_id)] += 1
            for j in range(d):
                sums[Int(best_id) * d + j] += x[r * d + j]
        var shift = Float32(0.0)
        for c in range(k):
            if counts[c] == 0:
                continue
            var norm2 = Float32(0.0)
            for j in range(d):
                var updated = sums[c * d + j] / Float32(counts[c])
                if metric == 0:
                    norm2 += updated * updated
                sums[c * d + j] = updated
            var invnorm = Float32(1.0)
            if metric == 0 and norm2 > 0.0:
                invnorm = 1.0 / sqrt(norm2)
            for j in range(d):
                var updated = sums[c * d + j] * invnorm
                var delta = updated - centers[c * d + j]
                shift += delta * delta
                centers[c * d + j] = updated
        if shift <= tol:
            break
    return objective


@export("mf_residuals")
def mf_residuals(
    x_addr: Int,
    centers_addr: Int,
    labels_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
) abi("C"):
    var x = fp(x_addr)
    var centers = fp(centers_addr)
    var labels = iptr(labels_addr)
    var dst = fp(dst_addr)
    for r in range(n):
        var center = Int(labels[r]) * d
        for j in range(d):
            dst[r * d + j] = x[r * d + j] - centers[center + j]


@export("mf_ivf_flat_search")
def mf_ivf_flat_search(
    vectors_addr: Int,
    vector_ids_addr: Int,
    offsets_addr: Int,
    queries_addr: Int,
    probes_addr: Int,
    dist_addr: Int,
    ids_addr: Int,
    nq: Int,
    d: Int,
    nprobe: Int,
    k: Int,
    metric: Int,
) abi("C"):
    var vectors = fp(vectors_addr)
    var vector_ids = iptr(vector_ids_addr)
    var offsets = iptr(offsets_addr)
    var queries = fp(queries_addr)
    var probes = iptr(probes_addr)
    var distances = fp(dist_addr)
    var ids = iptr(ids_addr)

    def search_query(q: Int) {imm}:
        var base = q * k
        init_topk(distances, ids, base, k, metric)
        for pidx in range(nprobe):
            var list_no = Int(probes[q * nprobe + pidx])
            var begin = Int(offsets[list_no])
            var end = Int(offsets[list_no + 1])
            for pos in range(begin, end):
                var value = score(queries + q * d, vectors + pos * d, d, metric)
                insert_topk(distances, ids, base, k, value, vector_ids[pos], metric)

    for q in range(nq):
        search_query(q)


@export("mf_pq_encode")
def mf_pq_encode(
    x_addr: Int,
    codebooks_addr: Int,
    codes_addr: Int,
    n: Int,
    d: Int,
    m: Int,
    ksub: Int,
) abi("C"):
    var x = fp(x_addr)
    var codebooks = fp(codebooks_addr)
    var codes = bp(codes_addr)
    var dsub = d // m

    def encode_row(r: Int) {imm}:
        for sub in range(m):
            var cb_base = (sub * ksub) * dsub
            var best_id = 0
            if dsub == PW:
                var xv = x.load[width=PW](r * d + sub * dsub)
                var delta = xv - codebooks.load[width=PW](cb_base)
                var best = (delta * delta).reduce_add()
                for c in range(1, ksub):
                    delta = xv - codebooks.load[width=PW](cb_base + c * dsub)
                    var value = (delta * delta).reduce_add()
                    if value < best:
                        best = value
                        best_id = c
            else:
                var best = pq_l2(
                    x + r * d + sub * dsub, codebooks + cb_base, dsub
                )
                for c in range(1, ksub):
                    var value = pq_l2(
                        x + r * d + sub * dsub,
                        codebooks + cb_base + c * dsub,
                        dsub,
                    )
                    if value < best:
                        best = value
                        best_id = c
            codes[r * m + sub] = UInt8(best_id)

    for r in range(n):
        encode_row(r)


@export("mf_pq_decode")
def mf_pq_decode(
    codes_addr: Int,
    codebooks_addr: Int,
    dst_addr: Int,
    n: Int,
    d: Int,
    m: Int,
    ksub: Int,
) abi("C"):
    var codes = bp(codes_addr)
    var codebooks = fp(codebooks_addr)
    var dst = fp(dst_addr)
    var dsub = d // m
    for r in range(n):
        for sub in range(m):
            var code = Int(codes[r * m + sub])
            var cb_base = (sub * ksub + code) * dsub
            for j in range(dsub):
                dst[r * d + sub * dsub + j] = codebooks[cb_base + j]


@export("mf_pq_tables")
def mf_pq_tables(
    queries_addr: Int,
    codebooks_addr: Int,
    tables_addr: Int,
    nq: Int,
    d: Int,
    m: Int,
    ksub: Int,
    metric: Int,
) abi("C"):
    var queries = fp(queries_addr)
    var codebooks = fp(codebooks_addr)
    var tables = fp(tables_addr)
    var dsub = d // m
    for q in range(nq):
        for sub in range(m):
            for c in range(ksub):
                tables[(q * m + sub) * ksub + c] = score(
                    queries + q * d + sub * dsub,
                    codebooks + (sub * ksub + c) * dsub,
                    dsub,
                    metric,
                )


@export("mf_pq_scan")
def mf_pq_scan(
    codes_addr: Int,
    vector_ids_addr: Int,
    tables_addr: Int,
    dist_addr: Int,
    ids_addr: Int,
    nb: Int,
    nq: Int,
    m: Int,
    ksub: Int,
    k: Int,
    metric: Int,
) abi("C"):
    var codes = bp(codes_addr)
    var vector_ids = iptr(vector_ids_addr)
    var tables = fp(tables_addr)
    var distances = fp(dist_addr)
    var ids = iptr(ids_addr)

    def scan_query(q: Int) {imm}:
        var base = q * k
        var table_base = q * m * ksub
        init_topk(distances, ids, base, k, metric)
        for r in range(nb):
            var value = Float32(0.0)
            if m == 8:
                value = tables[table_base + Int(codes[r * m])]
                value += tables[table_base + ksub + Int(codes[r * m + 1])]
                value += tables[table_base + 2 * ksub + Int(codes[r * m + 2])]
                value += tables[table_base + 3 * ksub + Int(codes[r * m + 3])]
                value += tables[table_base + 4 * ksub + Int(codes[r * m + 4])]
                value += tables[table_base + 5 * ksub + Int(codes[r * m + 5])]
                value += tables[table_base + 6 * ksub + Int(codes[r * m + 6])]
                value += tables[table_base + 7 * ksub + Int(codes[r * m + 7])]
            else:
                for sub in range(m):
                    value += tables[
                        table_base + sub * ksub + Int(codes[r * m + sub])
                    ]
            insert_topk(distances, ids, base, k, value, vector_ids[r], metric)

    for q in range(nq):
        scan_query(q)


@export("mf_ivfpq_search")
def mf_ivfpq_search(
    codes_addr: Int,
    vector_ids_addr: Int,
    offsets_addr: Int,
    queries_addr: Int,
    centers_addr: Int,
    probes_addr: Int,
    codebooks_addr: Int,
    tables_addr: Int,
    dist_addr: Int,
    ids_addr: Int,
    nq: Int,
    d: Int,
    m: Int,
    ksub: Int,
    nprobe: Int,
    k: Int,
    metric: Int,
) abi("C"):
    var codes = bp(codes_addr)
    var vector_ids = iptr(vector_ids_addr)
    var offsets = iptr(offsets_addr)
    var queries = fp(queries_addr)
    var centers = fp(centers_addr)
    var probes = iptr(probes_addr)
    var codebooks = fp(codebooks_addr)
    var tables = fp(tables_addr)
    var distances = fp(dist_addr)
    var ids = iptr(ids_addr)
    var dsub = d // m

    def search_query(q: Int) {imm}:
        var base = q * k
        init_topk(distances, ids, base, k, metric)
        for pidx in range(nprobe):
            var list_no = Int(probes[q * nprobe + pidx])
            var begin = Int(offsets[list_no])
            var end = Int(offsets[list_no + 1])
            var table_base = (q * nprobe + pidx) * m * ksub
            for sub in range(m):
                for c in range(ksub):
                    var tvalue = Float32(0.0)
                    if metric == 0:
                        for j in range(dsub):
                            var reconstructed = centers[
                                list_no * d + sub * dsub + j
                            ]
                            reconstructed += codebooks[
                                (sub * ksub + c) * dsub + j
                            ]
                            tvalue += (
                                queries[q * d + sub * dsub + j] * reconstructed
                            )
                    else:
                        var vacc = SIMD[DType.float32, PW](0.0)
                        var j = 0
                        while j + PW <= dsub:
                            var residual = (
                                queries.load[width=PW](q * d + sub * dsub + j)
                                - centers.load[width=PW](
                                    list_no * d + sub * dsub + j
                                )
                            )
                            var delta = residual - codebooks.load[width=PW](
                                (sub * ksub + c) * dsub + j
                            )
                            vacc += delta * delta
                            j += PW
                        tvalue = vacc.reduce_add()
                        while j < dsub:
                            var residual = queries[q * d + sub * dsub + j]
                            residual -= centers[list_no * d + sub * dsub + j]
                            var delta = (
                                residual
                                - codebooks[(sub * ksub + c) * dsub + j]
                            )
                            tvalue += delta * delta
                            j += 1
                    tables[table_base + sub * ksub + c] = tvalue
            for pos in range(begin, end):
                var value = Float32(0.0)
                if m == 8:
                    value = tables[table_base + Int(codes[pos * m])]
                    value += tables[
                        table_base + ksub + Int(codes[pos * m + 1])
                    ]
                    value += tables[
                        table_base + 2 * ksub + Int(codes[pos * m + 2])
                    ]
                    value += tables[
                        table_base + 3 * ksub + Int(codes[pos * m + 3])
                    ]
                    value += tables[
                        table_base + 4 * ksub + Int(codes[pos * m + 4])
                    ]
                    value += tables[
                        table_base + 5 * ksub + Int(codes[pos * m + 5])
                    ]
                    value += tables[
                        table_base + 6 * ksub + Int(codes[pos * m + 6])
                    ]
                    value += tables[
                        table_base + 7 * ksub + Int(codes[pos * m + 7])
                    ]
                else:
                    for sub in range(m):
                        value += tables[
                            table_base
                            + sub * ksub
                            + Int(codes[pos * m + sub])
                        ]
                insert_topk(distances, ids, base, k, value, vector_ids[pos], metric)

    for q in range(nq):
        search_query(q)
