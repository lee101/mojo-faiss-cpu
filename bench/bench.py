"""Benchmarks against faiss-cpu using identical data and trained parameters."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import faiss
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojofaiss as mf  # noqa: E402
from mojofaiss.indexes import _train_kmeans  # noqa: E402


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def vectors(n: int, d: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


CASES = []


def case(name):
    def decorate(fn):
        CASES.append((name, fn))
        return fn

    return decorate


@case("IndexFlatL2.search (20k x 200, d=64, k=10)")
def flat_l2():
    xb, xq = vectors(20_000, 64, 1), vectors(200, 64, 2)
    ours, upstream = mf.IndexFlatL2(64), faiss.IndexFlatL2(64)
    ours.add(xb)
    upstream.add(xb)
    return lambda: ours.search(xq, 10), lambda: upstream.search(xq, 10)


@case("IndexFlatIP.search (20k x 200, d=64, k=10)")
def flat_ip():
    xb, xq = vectors(20_000, 64, 3), vectors(200, 64, 4)
    ours, upstream = mf.IndexFlatIP(64), faiss.IndexFlatIP(64)
    ours.add(xb)
    upstream.add(xb)
    return lambda: ours.search(xq, 10), lambda: upstream.search(xq, 10)


@case("IndexIVFFlat.search (50k x 300, d=32, nprobe=8)")
def ivf_flat():
    xb, xq = vectors(50_000, 32, 5), vectors(300, 32, 6)
    centers, _, _ = _train_kmeans(xb[:5000], 128, niter=10)
    oq = mf.IndexFlatL2(32)
    oq.add(centers)
    ours = mf.IndexIVFFlat(oq, 32, 128)
    ours.is_trained = True
    tq = faiss.IndexFlatL2(32)
    tq.add(centers)
    upstream = faiss.IndexIVFFlat(tq, 32, 128)
    upstream.is_trained = True
    ours.add(xb)
    upstream.add(xb)
    ours.nprobe = upstream.nprobe = 8
    return lambda: ours.search(xq, 10), lambda: upstream.search(xq, 10)


def shared_pq(xtrain: np.ndarray, d: int, m: int):
    ours = mf.ProductQuantizer(d, m, 4)
    ours.niter = 10
    ours.train(xtrain)
    upstream = faiss.ProductQuantizer(d, m, 4)
    faiss.copy_array_to_vector(ours.centroids.ravel(), upstream.centroids)
    return ours, upstream


@case("IndexPQ.search (30k x 300, d=32, M=8, 4-bit)")
def pq_search():
    xb, xq = vectors(30_000, 32, 7), vectors(300, 32, 8)
    pq, _ = shared_pq(xb[:5000], 32, 8)
    ours = mf.IndexPQ(32, 8, 4)
    ours.pq = pq
    ours.is_trained = True
    upstream = faiss.IndexPQ(32, 8, 4)
    faiss.copy_array_to_vector(pq.centroids.ravel(), upstream.pq.centroids)
    upstream.is_trained = True
    ours.add(xb)
    upstream.add(xb)
    return lambda: ours.search(xq, 10), lambda: upstream.search(xq, 10)


@case("ProductQuantizer.compute_codes (50k, d=32, M=8, 4-bit)")
def pq_encode():
    x = vectors(50_000, 32, 9)
    ours, upstream = shared_pq(x[:5000], 32, 8)
    return lambda: ours.compute_codes(x), lambda: upstream.compute_codes(x)


@case("IndexIVFPQ.search (50k x 300, d=32, nprobe=8)")
def ivfpq_search():
    xb, xq = vectors(50_000, 32, 10), vectors(300, 32, 11)
    ours = mf.IndexIVFPQ(mf.IndexFlatL2(32), 32, 128, 8, 4)
    ours.pq.niter = 10
    ours.train(xb[:5000])
    tq = faiss.IndexFlatL2(32)
    tq.add(ours._centers)
    upstream = faiss.IndexIVFPQ(tq, 32, 128, 8, 4)
    faiss.copy_array_to_vector(ours.pq.centroids.ravel(), upstream.pq.centroids)
    upstream.is_trained = True
    ours.add(xb)
    upstream.add(xb)
    ours.nprobe = upstream.nprobe = 8
    return lambda: ours.search(xq, 10), lambda: upstream.search(xq, 10)


def main() -> None:
    print(f"Machine: {cpu_name()}")
    print(
        f"OS: {platform.system()} {platform.release()}; "
        f"faiss-cpu threads: {faiss.omp_get_max_threads()}; timing: best of 5"
    )
    print()
    print("| Operation | Mojo | faiss-cpu | faiss/Mojo | Result |")
    print("|---|---:|---:|---:|---|")
    for name, prepare in CASES:
        ours, upstream = prepare()
        if "compute_codes" in name:
            if not np.array_equal(ours(), upstream()):
                raise AssertionError(f"benchmark setup differs for {name}")
        else:
            got_d, got_i = ours()
            ref_d, ref_i = upstream()
            if not np.array_equal(got_i, ref_i):
                raise AssertionError(f"benchmark neighbor IDs differ for {name}")
            if not np.allclose(got_d, ref_d, rtol=4e-6, atol=5e-6):
                raise AssertionError(f"benchmark distances differ for {name}")
        mojo_time = timeit(ours)
        upstream_time = timeit(upstream)
        ratio = upstream_time / mojo_time
        result = "faster" if ratio > 1.0 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {ratio:.2f}x | {result} |"
        )


if __name__ == "__main__":
    main()
