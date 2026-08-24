"""Faiss-compatible flat, inverted-file, and product-quantized indexes."""

from __future__ import annotations

import copy
import operator
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ._lib import addr, f32, i64, lib

METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1
_THREAD_POOL_WORKERS = 32
_PARALLEL_WORK = 250_000
_EXECUTOR: ThreadPoolExecutor | None = None


def _parallel_ranges(n: int, work_size: int, max_workers: int, fn) -> None:
    global _EXECUTOR
    if n < 2 or work_size < _PARALLEL_WORK:
        fn(0, n)
        return
    workers = min(n, max_workers)
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=_THREAD_POOL_WORKERS)
    chunk = (n + workers - 1) // workers
    futures = [
        _EXECUTOR.submit(fn, start, min(start + chunk, n))
        for start in range(0, n, chunk)
    ]
    for future in futures:
        future.result()


def _integer(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _matrix(x, d: int, name: str = "x") -> np.ndarray:
    arr = f32(x)
    if arr.ndim != 2 or arr.shape[1] != d:
        raise ValueError(f"{name} must have shape (n, {d})")
    return arr


def _empty_search(n: int, k: int, metric: int) -> tuple[np.ndarray, np.ndarray]:
    sentinel = np.finfo(np.float32).max
    if metric == METRIC_INNER_PRODUCT:
        sentinel = -sentinel
    return (
        np.full((n, k), sentinel, dtype=np.float32),
        np.full((n, k), -1, dtype=np.int64),
    )


def _flat_search(
    xb: np.ndarray, xq: np.ndarray, k: int, metric: int
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(xq) == 0:
        return _empty_search(0, k, metric)
    if len(xb) == 0:
        return _empty_search(len(xq), k, metric)
    distances = np.empty((len(xq), k), dtype=np.float32)
    labels = np.empty((len(xq), k), dtype=np.int64)
    lib().mf_flat_search(
        addr(xb),
        addr(xq),
        addr(distances),
        addr(labels),
        len(xb),
        len(xq),
        xb.shape[1],
        k,
        metric,
    )
    return distances, labels


def _assign(x: np.ndarray, centers: np.ndarray, metric: int) -> np.ndarray:
    if len(centers) == 0:
        raise RuntimeError("cannot assign vectors without centroids")
    labels = np.empty(len(x), dtype=np.int64)
    if len(x):
        lib().mf_assign(
            addr(x), addr(centers), addr(labels), len(x), x.shape[1], len(centers), metric
        )
    return labels


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    nz = norms > 0
    x[nz] /= norms[nz, None]
    return x


def _kmeans_pp(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    if len(x) < k:
        raise RuntimeError(
            f"Number of training points ({len(x)}) should be at least as large "
            f"as number of clusters ({k})"
        )
    rng = np.random.default_rng(seed)
    centers = np.empty((k, x.shape[1]), dtype=np.float32)
    first = int(rng.integers(len(x)))
    centers[0] = x[first]
    best = np.sum((x - centers[0]) ** 2, axis=1, dtype=np.float32)
    for c in range(1, k):
        total = float(best.sum(dtype=np.float64))
        if total == 0.0:
            chosen = c % len(x)
        else:
            chosen = int(rng.choice(len(x), p=best.astype(np.float64) / total))
        centers[c] = x[chosen]
        candidate = np.sum((x - centers[c]) ** 2, axis=1, dtype=np.float32)
        np.minimum(best, candidate, out=best)
    return centers


def _train_kmeans(
    x: np.ndarray,
    k: int,
    *,
    metric: int = METRIC_L2,
    seed: int = 1234,
    niter: int = 25,
) -> tuple[np.ndarray, np.ndarray, float]:
    if niter <= 0:
        raise ValueError("niter must be positive")
    centers = _kmeans_pp(x, k, seed)
    if metric == METRIC_INNER_PRODUCT:
        _normalize_rows(centers)
    labels = np.empty(len(x), dtype=np.int64)
    sums = np.empty((k, x.shape[1]), dtype=np.float32)
    counts = np.empty(k, dtype=np.int64)
    objective = lib().mf_kmeans(
        addr(x),
        addr(centers),
        addr(labels),
        addr(sums),
        addr(counts),
        len(x),
        x.shape[1],
        k,
        niter,
        np.float32(1e-7),
        metric,
    )
    return centers, labels, float(objective)


def _pack_codes(codes: np.ndarray, nbits: int) -> np.ndarray:
    if nbits == 8:
        return np.ascontiguousarray(codes, dtype=np.uint8)
    n, m = codes.shape
    if nbits == 4 and m % 2 == 0:
        packed = np.empty((n, m // 2), dtype=np.uint8)
        np.left_shift(codes[:, 1::2], 4, out=packed)
        np.bitwise_or(packed, codes[:, 0::2], out=packed)
        return packed
    size = (m * nbits + 7) // 8
    packed = np.zeros((n, size), dtype=np.uint8)
    mask = (1 << nbits) - 1
    for sub in range(m):
        bit = sub * nbits
        byte, shift = divmod(bit, 8)
        values = codes[:, sub].astype(np.uint16) & mask
        packed[:, byte] |= (values << shift).astype(np.uint8)
        if shift + nbits > 8:
            packed[:, byte + 1] |= (values >> (8 - shift)).astype(np.uint8)
    return packed


def _unpack_codes(codes: np.ndarray, m: int, nbits: int) -> np.ndarray:
    packed = np.ascontiguousarray(codes, dtype=np.uint8)
    if packed.ndim != 2 or packed.shape[1] != (m * nbits + 7) // 8:
        raise ValueError("invalid packed-code shape")
    if nbits == 8:
        return packed.copy()
    unpacked = np.empty((len(packed), m), dtype=np.uint8)
    mask = (1 << nbits) - 1
    for sub in range(m):
        bit = sub * nbits
        byte, shift = divmod(bit, 8)
        values = packed[:, byte].astype(np.uint16) >> shift
        if shift + nbits > 8:
            values |= packed[:, byte + 1].astype(np.uint16) << (8 - shift)
        unpacked[:, sub] = (values & mask).astype(np.uint8)
    return unpacked


class Index:
    def __init__(self, d: int, metric: int = METRIC_L2):
        d = _integer(d, "dimension")
        metric = _integer(metric, "metric")
        if d <= 0:
            raise ValueError("dimension must be positive")
        if metric not in (METRIC_L2, METRIC_INNER_PRODUCT):
            raise ValueError("only METRIC_L2 and METRIC_INNER_PRODUCT are supported")
        self.d = d
        self.metric_type = metric
        self.is_trained = True
        self.verbose = False

    @property
    def ntotal(self) -> int:
        raise NotImplementedError

    def train(self, x) -> None:
        _matrix(x, self.d)

    def add(self, x) -> None:
        raise NotImplementedError

    def search(self, x, k: int, *, params=None) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def assign(self, x, k: int = 1) -> np.ndarray:
        return self.search(x, k)[1]

    def reset(self) -> None:
        raise NotImplementedError

    def reconstruct(self, key: int) -> np.ndarray:
        raise NotImplementedError

    def reconstruct_n(self, i0: int = 0, ni: int = -1) -> np.ndarray:
        i0 = _integer(i0, "i0")
        ni = _integer(ni, "ni")
        if ni < 0:
            ni = self.ntotal - i0
        if i0 < 0 or ni < 0 or i0 + ni > self.ntotal:
            raise RuntimeError("invalid reconstruction range")
        if ni == 0:
            return np.empty((0, self.d), dtype=np.float32)
        return np.vstack([self.reconstruct(i) for i in range(i0, i0 + ni)]).astype(
            np.float32, copy=False
        )

    def search_and_reconstruct(
        self, x, k: int, *, params=None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        distances, labels = self.search(x, k, params=params)
        recon = np.zeros((len(labels), k, self.d), dtype=np.float32)
        for q in range(len(labels)):
            for j in range(k):
                if labels[q, j] >= 0:
                    recon[q, j] = self.reconstruct(int(labels[q, j]))
        return distances, labels, recon


class IndexFlat(Index):
    def __init__(self, d: int, metric: int = METRIC_L2):
        super().__init__(d, metric)
        self._xb = np.empty((0, d), dtype=np.float32)
        self.code_size = 4 * d

    @property
    def ntotal(self) -> int:
        return len(self._xb)

    def add(self, x) -> None:
        x = _matrix(x, self.d)
        if len(x):
            self._xb = np.ascontiguousarray(np.vstack((self._xb, x)), dtype=np.float32)

    def search(self, x, k: int, *, params=None) -> tuple[np.ndarray, np.ndarray]:
        return _flat_search(
            self._xb, _matrix(x, self.d), _integer(k, "k"), self.metric_type
        )

    def range_search(self, x, thresh: float, *, params=None):
        x = _matrix(x, self.d)
        threshold = np.float32(thresh)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite and representable as float32")
        counts = np.empty(len(x), dtype=np.int64)
        if self.ntotal and len(x):
            lib().mf_range_count(
                addr(self._xb),
                addr(x),
                addr(counts),
                self.ntotal,
                len(x),
                self.d,
                threshold,
                self.metric_type,
            )
        else:
            counts.fill(0)
        lims = np.empty(len(x) + 1, dtype=np.int64)
        lims[0] = 0
        np.cumsum(counts, out=lims[1:])
        distances = np.empty(int(lims[-1]), dtype=np.float32)
        labels = np.empty(int(lims[-1]), dtype=np.int64)
        if len(distances):
            lib().mf_range_fill(
                addr(self._xb),
                addr(x),
                addr(lims),
                addr(distances),
                addr(labels),
                self.ntotal,
                len(x),
                self.d,
                threshold,
                self.metric_type,
            )
        return lims, distances, labels

    def reset(self) -> None:
        self._xb = np.empty((0, self.d), dtype=np.float32)

    def reconstruct(self, key: int) -> np.ndarray:
        if key < 0 or key >= self.ntotal:
            raise RuntimeError("invalid key")
        return self._xb[key].copy()

    def compute_distance_subset(self, x, labels):
        x = _matrix(x, self.d)
        labels = i64(labels)
        if labels.ndim != 2 or labels.shape[0] != len(x):
            raise ValueError("labels must have shape (nq, k)")
        if np.any(labels < 0) or np.any(labels >= self.ntotal):
            raise ValueError("labels contain an invalid database index")
        distances = np.empty(labels.shape, dtype=np.float32)
        for q in range(len(x)):
            selected = self._xb[labels[q]]
            if self.metric_type == METRIC_L2:
                distances[q] = np.sum((selected - x[q]) ** 2, axis=1)
            else:
                distances[q] = selected @ x[q]
        return distances


class IndexFlatL2(IndexFlat):
    def __init__(self, d: int):
        super().__init__(d, METRIC_L2)


class IndexFlatIP(IndexFlat):
    def __init__(self, d: int):
        super().__init__(d, METRIC_INNER_PRODUCT)


class ProductQuantizer:
    def __init__(self, d: int, M: int, nbits: int = 8):
        d = _integer(d, "dimension")
        M = _integer(M, "M")
        nbits = _integer(nbits, "nbits")
        if d <= 0 or M <= 0 or d % M:
            raise RuntimeError("dimension must be a positive multiple of M")
        if nbits < 1 or nbits > 8:
            raise RuntimeError("this port supports 1 through 8 bits per subquantizer")
        self.d = d
        self.M = M
        self.nbits = nbits
        self.dsub = d // M
        self.ksub = 1 << nbits
        self.code_size = (M * nbits + 7) // 8
        self.is_trained = False
        self.verbose = False
        self.niter = 25
        self.seed = 1234
        self.centroids = np.empty((M, self.ksub, self.dsub), dtype=np.float32)

    def train(self, x) -> None:
        x = _matrix(x, self.d)
        for sub in range(self.M):
            part = np.ascontiguousarray(
                x[:, sub * self.dsub : (sub + 1) * self.dsub], dtype=np.float32
            )
            centers, _, _ = _train_kmeans(
                part, self.ksub, seed=self.seed + sub, niter=self.niter
            )
            self.centroids[sub] = centers
        self.is_trained = True

    def _encode_unpacked(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("ProductQuantizer is not trained")
        codes = np.empty((len(x), self.M), dtype=np.uint8)
        def encode(start: int, end: int) -> None:
            lib().mf_pq_encode(
                addr(x[start:end]),
                addr(self.centroids),
                addr(codes[start:end]),
                end - start,
                self.d,
                self.M,
                self.ksub,
            )

        if len(x):
            _parallel_ranges(len(x), len(x) * self.d * self.ksub, 8, encode)
        return codes

    def compute_codes(self, x) -> np.ndarray:
        return _pack_codes(self._encode_unpacked(_matrix(x, self.d)), self.nbits)

    def decode(self, codes) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("ProductQuantizer is not trained")
        unpacked = _unpack_codes(codes, self.M, self.nbits)
        decoded = np.empty((len(unpacked), self.d), dtype=np.float32)
        if len(unpacked):
            lib().mf_pq_decode(
                addr(unpacked),
                addr(self.centroids),
                addr(decoded),
                len(unpacked),
                self.d,
                self.M,
                self.ksub,
            )
        return decoded


class IndexPQ(Index):
    def __init__(
        self, d: int, M: int, nbits: int = 8, metric: int = METRIC_L2
    ):
        super().__init__(d, metric)
        self.pq = ProductQuantizer(d, M, nbits)
        self.is_trained = False
        self.code_size = self.pq.code_size
        self._codes = np.empty((0, M), dtype=np.uint8)
        self._ids = np.empty(0, dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return len(self._codes)

    def train(self, x) -> None:
        self.pq.train(x)
        self.is_trained = True

    def add(self, x) -> None:
        x = _matrix(x, self.d)
        if not self.is_trained:
            raise RuntimeError("Index not trained")
        new = self.pq._encode_unpacked(x)
        self._codes = np.ascontiguousarray(np.vstack((self._codes, new)), dtype=np.uint8)
        self._ids = np.arange(len(self._codes), dtype=np.int64)

    def search(self, x, k: int, *, params=None) -> tuple[np.ndarray, np.ndarray]:
        x = _matrix(x, self.d)
        k = _integer(k, "k")
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.ntotal:
            return _empty_search(len(x), k, self.metric_type)
        if len(x) == 0:
            return _empty_search(0, k, self.metric_type)
        tables = np.empty((len(x), self.pq.M, self.pq.ksub), dtype=np.float32)
        distances = np.empty((len(x), k), dtype=np.float32)
        labels = np.empty((len(x), k), dtype=np.int64)
        def search_rows(start: int, end: int) -> None:
            count = end - start
            lib().mf_pq_tables(
                addr(x[start:end]),
                addr(self.pq.centroids),
                addr(tables[start:end]),
                count,
                self.d,
                self.pq.M,
                self.pq.ksub,
                self.metric_type,
            )
            lib().mf_pq_scan(
                addr(self._codes),
                addr(self._ids),
                addr(tables[start:end]),
                addr(distances[start:end]),
                addr(labels[start:end]),
                self.ntotal,
                count,
                self.pq.M,
                self.pq.ksub,
                k,
                self.metric_type,
            )

        _parallel_ranges(len(x), len(x) * self.ntotal * self.pq.M, 16, search_rows)
        return distances, labels

    def sa_encode(self, x) -> np.ndarray:
        return self.pq.compute_codes(x)

    def sa_decode(self, codes) -> np.ndarray:
        return self.pq.decode(codes)

    def reset(self) -> None:
        self._codes = np.empty((0, self.pq.M), dtype=np.uint8)
        self._ids = np.empty(0, dtype=np.int64)

    def reconstruct(self, key: int) -> np.ndarray:
        if key < 0 or key >= self.ntotal:
            raise RuntimeError("invalid key")
        packed = _pack_codes(self._codes[key : key + 1], self.pq.nbits)
        return self.pq.decode(packed)[0]


class _IVFBase(Index):
    def __init__(self, quantizer: Index, d: int, nlist: int, metric: int):
        super().__init__(d, metric)
        if quantizer.d != d:
            raise RuntimeError("quantizer dimension does not match index")
        if quantizer.metric_type != metric:
            raise RuntimeError("quantizer metric does not match index metric")
        nlist = _integer(nlist, "nlist")
        if nlist <= 0:
            raise ValueError("nlist must be positive")
        self.quantizer = quantizer
        self.nlist = nlist
        self.nprobe = 1
        self.is_trained = False
        self.own_fields = False
        self._next_id = 0
        self._list_nos = np.empty(0, dtype=np.int64)
        self._ids = np.empty(0, dtype=np.int64)
        self._packed_cache: Any = None

    @property
    def ntotal(self) -> int:
        return len(self._ids)

    @property
    def _centers(self) -> np.ndarray:
        if not isinstance(self.quantizer, IndexFlat):
            raise RuntimeError("this port requires an IndexFlat coarse quantizer")
        return self.quantizer._xb

    def train(self, x) -> None:
        x = _matrix(x, self.d)
        centers, _, _ = _train_kmeans(
            x, self.nlist, metric=self.metric_type, seed=1234, niter=25
        )
        self.quantizer.reset()
        self.quantizer.add(centers)
        self.quantizer.is_trained = True
        self.is_trained = True

    def _coarse_assign(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Index not trained")
        if len(self._centers) != self.nlist:
            raise RuntimeError("coarse quantizer must contain exactly nlist centroids")
        return _assign(x, self._centers, self.metric_type)

    def _append_ids(self, list_nos: np.ndarray, ids) -> np.ndarray:
        if ids is None:
            new_ids = np.arange(
                self._next_id, self._next_id + len(list_nos), dtype=np.int64
            )
        else:
            new_ids = i64(ids)
            if new_ids.ndim != 1:
                raise ValueError("ids must be a one-dimensional int64 array")
            if len(new_ids) != len(list_nos):
                raise ValueError("ids must contain one id per vector")
        if len(new_ids) and new_ids.max() == np.iinfo(np.int64).max:
            raise OverflowError("id is too large to advance the automatic id counter")
        self._next_id = max(
            self._next_id + len(list_nos),
            int(new_ids.max()) + 1 if len(new_ids) else 0,
        )
        self._list_nos = np.ascontiguousarray(
            np.concatenate((self._list_nos, list_nos)), dtype=np.int64
        )
        self._ids = np.ascontiguousarray(np.concatenate((self._ids, new_ids)), dtype=np.int64)
        self._packed_cache = None
        return new_ids

    def _probes(self, x: np.ndarray) -> np.ndarray:
        nprobe = min(max(_integer(self.nprobe, "nprobe"), 1), self.nlist)
        return self.quantizer.search(x, nprobe)[1]

    def reset(self) -> None:
        self._list_nos = np.empty(0, dtype=np.int64)
        self._ids = np.empty(0, dtype=np.int64)
        self._next_id = 0
        self._packed_cache = None


class IndexIVFFlat(_IVFBase):
    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        metric: int = METRIC_L2,
    ):
        super().__init__(quantizer, d, nlist, metric)
        self.code_size = 4 * d
        self._vectors = np.empty((0, d), dtype=np.float32)

    def add(self, x) -> None:
        self.add_with_ids(x, None)

    def add_with_ids(self, x, ids) -> None:
        x = _matrix(x, self.d)
        list_nos = self._coarse_assign(x)
        self._vectors = np.ascontiguousarray(
            np.vstack((self._vectors, x)), dtype=np.float32
        )
        self._append_ids(list_nos, ids)

    def _packed(self):
        if self._packed_cache is None:
            order = np.argsort(self._list_nos, kind="stable")
            counts = np.bincount(self._list_nos, minlength=self.nlist)
            offsets = np.empty(self.nlist + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            self._packed_cache = (
                np.ascontiguousarray(self._vectors[order], dtype=np.float32),
                np.ascontiguousarray(self._ids[order], dtype=np.int64),
                offsets,
            )
        return self._packed_cache

    def search(self, x, k: int, *, params=None) -> tuple[np.ndarray, np.ndarray]:
        x = _matrix(x, self.d)
        k = _integer(k, "k")
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.ntotal:
            return _empty_search(len(x), k, self.metric_type)
        if len(x) == 0:
            return _empty_search(0, k, self.metric_type)
        vectors, vector_ids, offsets = self._packed()
        probes = self._probes(x)
        distances = np.empty((len(x), k), dtype=np.float32)
        labels = np.empty((len(x), k), dtype=np.int64)
        def search_rows(start: int, end: int) -> None:
            lib().mf_ivf_flat_search(
                addr(vectors),
                addr(vector_ids),
                addr(offsets),
                addr(x[start:end]),
                addr(probes[start:end]),
                addr(distances[start:end]),
                addr(labels[start:end]),
                end - start,
                self.d,
                probes.shape[1],
                k,
                self.metric_type,
            )

        estimated = len(x) * self.ntotal * probes.shape[1] * self.d // self.nlist
        _parallel_ranges(len(x), estimated, 8, search_rows)
        return distances, labels

    def reset(self) -> None:
        super().reset()
        self._vectors = np.empty((0, self.d), dtype=np.float32)

    def reconstruct(self, key: int) -> np.ndarray:
        hits = np.flatnonzero(self._ids == key)
        if not len(hits):
            raise RuntimeError("invalid key")
        return self._vectors[hits[0]].copy()


class IndexIVFPQ(_IVFBase):
    def __init__(
        self,
        quantizer: Index,
        d: int,
        nlist: int,
        M: int,
        nbits: int = 8,
        metric: int = METRIC_L2,
    ):
        super().__init__(quantizer, d, nlist, metric)
        self.pq = ProductQuantizer(d, M, nbits)
        self.code_size = self.pq.code_size
        self.by_residual = True
        self._codes = np.empty((0, M), dtype=np.uint8)

    def train(self, x) -> None:
        x = _matrix(x, self.d)
        super().train(x)
        labels = self._coarse_assign(x)
        residuals = np.empty_like(x)
        lib().mf_residuals(
            addr(x), addr(self._centers), addr(labels), addr(residuals), len(x), self.d
        )
        self.pq.train(residuals)

    def add(self, x) -> None:
        self.add_with_ids(x, None)

    def add_with_ids(self, x, ids) -> None:
        x = _matrix(x, self.d)
        list_nos = self._coarse_assign(x)
        residuals = np.empty_like(x)
        if len(x):
            lib().mf_residuals(
                addr(x),
                addr(self._centers),
                addr(list_nos),
                addr(residuals),
                len(x),
                self.d,
            )
        codes = self.pq._encode_unpacked(residuals)
        self._codes = np.ascontiguousarray(np.vstack((self._codes, codes)), dtype=np.uint8)
        self._append_ids(list_nos, ids)

    def _packed(self):
        if self._packed_cache is None:
            order = np.argsort(self._list_nos, kind="stable")
            counts = np.bincount(self._list_nos, minlength=self.nlist)
            offsets = np.empty(self.nlist + 1, dtype=np.int64)
            offsets[0] = 0
            np.cumsum(counts, out=offsets[1:])
            self._packed_cache = (
                np.ascontiguousarray(self._codes[order], dtype=np.uint8),
                np.ascontiguousarray(self._ids[order], dtype=np.int64),
                offsets,
            )
        return self._packed_cache

    def search(self, x, k: int, *, params=None) -> tuple[np.ndarray, np.ndarray]:
        x = _matrix(x, self.d)
        k = _integer(k, "k")
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.ntotal:
            return _empty_search(len(x), k, self.metric_type)
        if len(x) == 0:
            return _empty_search(0, k, self.metric_type)
        codes, vector_ids, offsets = self._packed()
        probes = self._probes(x)
        tables = np.empty(
            (len(x), probes.shape[1], self.pq.M, self.pq.ksub), dtype=np.float32
        )
        distances = np.empty((len(x), k), dtype=np.float32)
        labels = np.empty((len(x), k), dtype=np.int64)
        def search_rows(start: int, end: int) -> None:
            lib().mf_ivfpq_search(
                addr(codes),
                addr(vector_ids),
                addr(offsets),
                addr(x[start:end]),
                addr(self._centers),
                addr(probes[start:end]),
                addr(self.pq.centroids),
                addr(tables[start:end]),
                addr(distances[start:end]),
                addr(labels[start:end]),
                end - start,
                self.d,
                self.pq.M,
                self.pq.ksub,
                probes.shape[1],
                k,
                self.metric_type,
            )

        table_work = probes.shape[1] * self.pq.ksub * self.d
        scan_work = self.ntotal * probes.shape[1] * self.pq.M // self.nlist
        _parallel_ranges(len(x), len(x) * (table_work + scan_work), 32, search_rows)
        return distances, labels

    def reset(self) -> None:
        super().reset()
        self._codes = np.empty((0, self.pq.M), dtype=np.uint8)

    def reconstruct(self, key: int) -> np.ndarray:
        hits = np.flatnonzero(self._ids == key)
        if not len(hits):
            raise RuntimeError("invalid key")
        pos = int(hits[0])
        packed = _pack_codes(self._codes[pos : pos + 1], self.pq.nbits)
        return self._centers[self._list_nos[pos]] + self.pq.decode(packed)[0]


class IndexIDMap(Index):
    def __init__(self, index: Index):
        super().__init__(index.d, index.metric_type)
        self.index = index
        self.id_map = np.empty(0, dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def is_trained(self):
        return self.index.is_trained

    @is_trained.setter
    def is_trained(self, value):
        if hasattr(self, "index"):
            self.index.is_trained = value

    def train(self, x) -> None:
        self.index.train(x)

    def add(self, x) -> None:
        raise RuntimeError("add does not make sense with IndexIDMap, use add_with_ids")

    def add_with_ids(self, x, ids) -> None:
        ids = i64(ids)
        if ids.ndim != 1:
            raise ValueError("ids must be a one-dimensional int64 array")
        x = _matrix(x, self.d)
        if len(ids) != len(x):
            raise ValueError("ids must contain one id per vector")
        self.index.add(x)
        self.id_map = np.concatenate((self.id_map, ids)).astype(np.int64, copy=False)

    def search(self, x, k: int, *, params=None):
        distances, internal = self.index.search(x, k, params=params)
        labels = np.full_like(internal, -1)
        valid = internal >= 0
        labels[valid] = self.id_map[internal[valid]]
        return distances, labels

    def reset(self) -> None:
        self.index.reset()
        self.id_map = np.empty(0, dtype=np.int64)

    def reconstruct(self, key: int) -> np.ndarray:
        hits = np.flatnonzero(self.id_map == key)
        if not len(hits):
            raise RuntimeError("invalid key")
        return self.index.reconstruct(int(hits[0]))


IndexIDMap2 = IndexIDMap


class Kmeans:
    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        verbose: bool = False,
        seed: int = 1234,
        spherical: bool = False,
        **kwargs,
    ):
        self.d = _integer(d, "dimension")
        self.k = _integer(k, "k")
        self.niter = _integer(niter, "niter")
        if self.d <= 0 or self.k <= 0 or self.niter <= 0:
            raise ValueError("dimension, k, and niter must be positive")
        self.verbose = bool(verbose)
        self.seed = _integer(seed, "seed")
        self.spherical = bool(spherical)
        self.centroids: np.ndarray | None = None
        self.obj: np.ndarray | None = None
        self.index: IndexFlat | None = None

    def train(self, x, weights=None, init_centroids=None):
        if weights is not None or init_centroids is not None:
            raise NotImplementedError("weights and init_centroids are not covered")
        x = _matrix(x, self.d)
        metric = METRIC_INNER_PRODUCT if self.spherical else METRIC_L2
        centers, _, objective = _train_kmeans(
            x, self.k, metric=metric, seed=self.seed, niter=self.niter
        )
        self.centroids = centers
        self.obj = np.array([objective], dtype=np.float32)
        self.index = IndexFlat(self.d, metric)
        self.index.add(centers)
        return objective

    def assign(self, x):
        if self.index is None:
            raise RuntimeError("Kmeans has not been trained")
        return self.index.search(x, 1)


def index_factory(d: int, description: str, metric: int = METRIC_L2) -> Index:
    spec = re.sub(r"\s+", "", description)
    quantizer = IndexFlat(d, metric)
    if spec == "Flat":
        return quantizer
    match = re.fullmatch(r"PQ(\d+)(?:x(\d+))?", spec)
    if match:
        return IndexPQ(d, int(match.group(1)), int(match.group(2) or 8), metric)
    match = re.fullmatch(r"IVF(\d+),Flat", spec)
    if match:
        return IndexIVFFlat(quantizer, d, int(match.group(1)), metric)
    match = re.fullmatch(r"IVF(\d+),PQ(\d+)(?:x(\d+))?", spec)
    if match:
        return IndexIVFPQ(
            quantizer,
            d,
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3) or 8),
            metric,
        )
    raise RuntimeError(f"unsupported index_factory description: {description!r}")


def clone_index(index: Index) -> Index:
    return copy.deepcopy(index)


def knn(xq, xb, k: int, metric: int = METRIC_L2):
    xb = f32(xb)
    if xb.ndim != 2:
        raise ValueError("xb must be a matrix")
    return _flat_search(
        xb, _matrix(xq, xb.shape[1], "xq"), _integer(k, "k"), metric
    )


def normalize_L2(x) -> None:
    if not isinstance(x, np.ndarray) or x.dtype != np.float32 or not x.flags.c_contiguous:
        raise TypeError("x must be a C-contiguous float32 NumPy array")
    if x.ndim != 2:
        raise ValueError("x must be a matrix")
    _normalize_rows(x)
