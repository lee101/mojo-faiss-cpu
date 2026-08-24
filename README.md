# mojo-faiss-cpu

`mojo-faiss-cpu` is an open-source CPU implementation of Faiss's core flat,
inverted-file, and product-quantized vector indexes. Compute-heavy distance,
assignment, encoding, and scan loops are compiled from Mojo into one shared
library; a small Python layer owns index state and presents familiar Faiss
class names and method signatures.

The import name is `mojofaiss`, allowing it and upstream `faiss-cpu` to be
loaded in the same process:

```python
import mojofaiss as faiss
```

## Coverage

Implemented:

- `IndexFlat`, `IndexFlatL2`, and `IndexFlatIP`: `add`, `search`,
  `range_search`, `assign`, `reconstruct`, `reconstruct_n`,
  `search_and_reconstruct`, `compute_distance_subset`, and `reset`
- `IndexIVFFlat`: training, `add`, `add_with_ids`, `search`, reconstruction,
  `nlist`, and `nprobe`
- `ProductQuantizer` and `IndexPQ`: 1- through 8-bit subquantizers, Faiss
  bit-packed public codes, training, encoding, decoding, search, and
  reconstruction
- residual `IndexIVFPQ`: training, `add`, `add_with_ids`, search, and
  reconstruction
- `IndexIDMap`/`IndexIDMap2`, `Kmeans`, `index_factory`, `clone_index`, `knn`,
  and `normalize_L2`
- squared L2 and inner-product metrics, with row-major `float32` vectors

Inputs crossing the compiled boundary must be C-contiguous after layout
normalization and must already have the documented dtype: vectors are
`float32`, while IDs and label matrices are `int64`. Other dtypes are rejected
instead of being silently narrowed.

This is deliberately not a complete Faiss replacement. It does not cover GPU
indexes, HNSW, scalar quantization, binary indexes, pre-transform indexes,
on-disk inverted lists, Faiss binary serialization, range search on IVF
indexes, ID selectors, removal, or advanced search parameters. IVF currently
requires an `IndexFlat` coarse quantizer. Mojo training is deterministic
k-means++ followed by Lloyd iterations; it is not expected to produce the same
centroids as Faiss's trainer from only the same seed.

Parity tests avoid hiding that training difference: trained centroids and
codebooks are shared between implementations, after which exact neighbor IDs,
packed codes, and numerically equivalent distances are asserted. Separately,
full-probe IVF trained entirely by this project is checked against exact
upstream flat search.

## Install and run

The repository pins the tested Mojo nightly and obtains `faiss-cpu`, NumPy,
and pytest from conda-forge:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-faiss-cpu.so`. Importing `mojofaiss`
also rebuilds a missing or stale library when a Mojo compiler is available.

This example trains and searches a residual IVF-PQ index:

```python
import numpy as np
import mojofaiss as faiss

rng = np.random.default_rng(0)
d = 32
xb = rng.normal(size=(10_000, d)).astype(np.float32)
xq = rng.normal(size=(5, d)).astype(np.float32)

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, 64, 8, 4)
index.train(xb[:4096])
index.add(xb)
index.nprobe = 8

distances, ids = index.search(xq, 10)
print(distances.shape, ids.shape)
```

Run it from the environment with `pixi run python example.py`.

## Benchmarks

These are real best-of-five results from `pixi run bench`. The ratio is
`faiss-cpu time / Mojo time`, so values above 1 mean this project was faster.
Before timing, the benchmark checks that both implementations return identical
neighbor IDs or codes and close distances. Faiss used its default 72 threads;
large independent query and encode ranges use 8, 16, or 32 workers according to
the kernel only after a work-size threshold, and stay serial for small inputs.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux
6.8.0-136-generic; faiss-cpu reported 72 threads.

| Operation | Mojo | faiss-cpu | faiss/Mojo | Result |
|---|---:|---:|---:|---|
| IndexFlatL2.search (20k x 200, d=64, k=10) | 57.32 ms | 1061.91 ms | 18.53x | faster |
| IndexFlatIP.search (20k x 200, d=64, k=10) | 56.02 ms | 667.74 ms | 11.92x | faster |
| IndexIVFFlat.search (50k x 300, d=32, nprobe=8) | 5.78 ms | 1.96 ms | 0.34x | slower |
| IndexPQ.search (30k x 300, d=32, M=8, 4-bit) | 9.79 ms | 17.24 ms | 1.76x | faster |
| ProductQuantizer.compute_codes (50k, d=32, M=8, 4-bit) | 6.20 ms | 1.11 ms | 0.18x | slower |
| IndexIVFPQ.search (50k x 300, d=32, nprobe=8) | 12.76 ms | 2.00 ms | 0.16x | slower |

The flat workload benefits from a compact single-threaded SIMD scan and avoids
the overhead Faiss incurs with its 72-thread default on this query batch. PQ
search is also faster on this run. IVF-flat, PQ encoding, and IVF-PQ remain
slower than Faiss's mature specialized kernels. No GPU path was added: the
distance loops perform about 0.5 FLOP per byte loaded and PQ scanning is lower,
so they do not meet the roughly 2 FLOP/byte threshold needed to justify device
transfer and launch overhead.

## How it works

Python passes NumPy buffer addresses and integer extents through `ctypes`.
Every exported Mojo function has a non-parametric C ABI and reconstructs
`UnsafePointer` values from those addresses. NumPy owns all allocations, so the
shared library neither retains Python pointers nor exposes an allocator across
the FFI boundary. Large row ranges are split into zero-copy NumPy views and
dispatched concurrently; each worker calls the same compiled SIMD kernel, while
small ranges take one direct serial call.

Vectors and centroids are contiguous row-major `float32`; labels and offsets
are `int64`. Flat L2 and inner-product reductions use eight-lane SIMD. PQ
encoding and residual tables use a native four-float SIMD stage plus a scalar
tail. IVF vectors or codes are stably grouped by list and scanned through an
`int64` offset table. PQ distance tables are built once per query for
`IndexPQ`; IVF-PQ builds a residual table per selected list instead of
reconstructing every candidate dimension. Public PQ codes use Faiss's packed
bit layout, with a direct allocation-light 4-bit packing path, while scans use
an internal byte per subquantizer for direct lookup.

All kernels live in `src/capi.mojo` so the fixed Mojo shared-library build cost
is paid once.
