import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

import mojofaiss as mf
from mojofaiss.indexes import _pack_codes, _train_kmeans, _unpack_codes


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(600, 24)).astype(np.float32)
    query = rng.normal(size=(23, 24)).astype(np.float32)
    return train, query


def _shared_ivf_flat(x, nlist=12):
    centers, _, _ = _train_kmeans(x, nlist, niter=12)
    ours_q = mf.IndexFlatL2(x.shape[1])
    ours_q.add(centers)
    ours = mf.IndexIVFFlat(ours_q, x.shape[1], nlist)
    ours.is_trained = True
    upstream_q = faiss.IndexFlatL2(x.shape[1])
    upstream_q.add(centers)
    upstream = faiss.IndexIVFFlat(upstream_q, x.shape[1], nlist)
    upstream.is_trained = True
    return ours, upstream


@pytest.mark.parametrize("nprobe", [1, 4, 12])
def test_ivf_flat_search_with_shared_training_parity(data, nprobe):
    x, q = data
    ours, upstream = _shared_ivf_flat(x)
    ours.add(x)
    upstream.add(x)
    ours.nprobe = upstream.nprobe = nprobe
    got_d, got_i = ours.search(q, 15)
    ref_d, ref_i = upstream.search(q, 15)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=3e-6)


def test_ivf_flat_external_ids_parity(data):
    x, q = data
    ids = np.arange(len(x), dtype=np.int64) * 11 + 3
    ours, upstream = _shared_ivf_flat(x)
    ours.add_with_ids(x, ids)
    upstream.add_with_ids(x, ids)
    ours.nprobe = upstream.nprobe = 5
    got_d, got_i = ours.search(q, 11)
    ref_d, ref_i = upstream.search(q, 11)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=3e-6)
    assert np.array_equal(ours.reconstruct(int(ids[10])), x[10])


def test_ivf_training_has_exact_full_probe_results(data):
    x, q = data
    index = mf.IndexIVFFlat(mf.IndexFlatL2(24), 24, 12)
    index.train(x)
    index.add(x)
    index.nprobe = index.nlist
    got_d, got_i = index.search(q, 10)
    flat = faiss.IndexFlatL2(24)
    flat.add(x)
    ref_d, ref_i = flat.search(q, 10)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=3e-6)


@pytest.mark.parametrize(
    ("m", "nbits"),
    [(6, 1), (6, 2), (4, 3), (6, 4), (3, 5), (3, 6), (3, 7), (3, 8)],
)
def test_pq_packing_matches_upstream(data, m, nbits):
    x, _ = data
    pq = mf.ProductQuantizer(24, m, nbits)
    pq.niter = 8
    pq.train(x)
    upstream = faiss.ProductQuantizer(24, m, nbits)
    faiss.copy_array_to_vector(pq.centroids.ravel(), upstream.centroids)
    got = pq.compute_codes(x[:40])
    ref = upstream.compute_codes(x[:40])
    assert np.array_equal(got, ref)
    assert np.array_equal(_pack_codes(_unpack_codes(got, m, nbits), nbits), got)
    assert np.allclose(pq.decode(got), upstream.decode(ref))


def test_pq_simd_tail_and_parallel_encode_parity():
    x = np.random.default_rng(31).normal(size=(4500, 14)).astype(np.float32)
    ours = mf.ProductQuantizer(14, 2, 4)
    ours.niter = 6
    ours.train(x[:600])
    upstream = faiss.ProductQuantizer(14, 2, 4)
    faiss.copy_array_to_vector(ours.centroids.ravel(), upstream.centroids)
    assert np.array_equal(ours.compute_codes(x), upstream.compute_codes(x))


def test_index_pq_search_with_shared_codebooks_parity(data):
    x, q = data
    ours = mf.IndexPQ(24, 6, 4)
    ours.pq.niter = 10
    ours.train(x)
    upstream = faiss.IndexPQ(24, 6, 4)
    faiss.copy_array_to_vector(ours.pq.centroids.ravel(), upstream.pq.centroids)
    upstream.is_trained = True
    ours.add(x)
    upstream.add(x)
    got_d, got_i = ours.search(q, 17)
    ref_d, ref_i = upstream.search(q, 17)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=3e-6)
    code = ours.sa_encode(x[:1])
    assert np.allclose(ours.reconstruct(0), upstream.sa_decode(code)[0])


def test_parallel_pq_search_threshold_parity(data):
    x, q = data
    q = np.tile(q, (5, 1))
    ours = mf.IndexPQ(24, 6, 4)
    ours.pq.niter = 8
    ours.train(x)
    upstream = faiss.IndexPQ(24, 6, 4)
    faiss.copy_array_to_vector(ours.pq.centroids.ravel(), upstream.pq.centroids)
    upstream.is_trained = True
    ours.add(x)
    upstream.add(x)
    got_d, got_i = ours.search(q, 11)
    ref_d, ref_i = upstream.search(q, 11)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=3e-6)


def test_ivfpq_search_with_shared_training_parity(data):
    x, q = data
    ours = mf.IndexIVFPQ(mf.IndexFlatL2(24), 24, 12, 6, 4)
    ours.pq.niter = 10
    ours.train(x)
    upstream_q = faiss.IndexFlatL2(24)
    upstream_q.add(ours._centers)
    upstream = faiss.IndexIVFPQ(upstream_q, 24, 12, 6, 4)
    faiss.copy_array_to_vector(ours.pq.centroids.ravel(), upstream.pq.centroids)
    upstream.is_trained = True
    ours.add(x)
    upstream.add(x)
    ours.nprobe = upstream.nprobe = 5
    got_d, got_i = ours.search(q, 15)
    ref_d, ref_i = upstream.search(q, 15)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=3e-6, atol=4e-6)


def test_ivfpq_external_ids_reconstruct_and_reset(data):
    x, _ = data
    ids = np.arange(80, dtype=np.int64) * 13 + 5
    index = mf.IndexIVFPQ(mf.IndexFlatL2(24), 24, 12, 6, 4)
    index.pq.niter = 6
    index.train(x)
    index.add_with_ids(x[:80], ids)
    reconstructed = index.reconstruct(int(ids[7]))
    assert reconstructed.shape == (24,)
    assert reconstructed.dtype == np.float32
    index.reset()
    assert index.ntotal == 0


def test_index_pq_reset_and_empty_query(data):
    x, _ = data
    index = mf.IndexPQ(24, 6, 4)
    index.pq.niter = 6
    index.train(x)
    index.add(x[:30])
    distances, labels = index.search(np.empty((0, 24), dtype=np.float32), 5)
    assert distances.shape == labels.shape == (0, 5)
    index.reset()
    assert index.ntotal == 0


def test_parallel_ivf_search_thresholds_parity(data):
    x, q = data
    q = np.tile(q, (8, 1))

    ours_flat, upstream_flat = _shared_ivf_flat(x)
    ours_flat.add(x)
    upstream_flat.add(x)
    ours_flat.nprobe = upstream_flat.nprobe = 12
    got_d, got_i = ours_flat.search(q, 9)
    ref_d, ref_i = upstream_flat.search(q, 9)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=3e-6)

    ours_pq = mf.IndexIVFPQ(mf.IndexFlatL2(24), 24, 12, 6, 4)
    ours_pq.pq.niter = 8
    ours_pq.train(x)
    upstream_q = faiss.IndexFlatL2(24)
    upstream_q.add(ours_pq._centers)
    upstream_pq = faiss.IndexIVFPQ(upstream_q, 24, 12, 6, 4)
    faiss.copy_array_to_vector(
        ours_pq.pq.centroids.ravel(), upstream_pq.pq.centroids
    )
    upstream_pq.is_trained = True
    ours_pq.add(x)
    upstream_pq.add(x)
    ours_pq.nprobe = upstream_pq.nprobe = 12
    got_d, got_i = ours_pq.search(q, 9)
    ref_d, ref_i = upstream_pq.search(q, 9)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=3e-6, atol=4e-6)


@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        ("Flat", mf.IndexFlat),
        ("IVF12,Flat", mf.IndexIVFFlat),
        ("PQ6x4", mf.IndexPQ),
        ("IVF12,PQ6x4", mf.IndexIVFPQ),
    ],
)
def test_index_factory(spec, kind):
    index = mf.index_factory(24, spec)
    assert isinstance(index, kind)
    assert index.d == 24


def test_kmeans_quality_against_upstream(data):
    x, _ = data
    ours = mf.Kmeans(24, 10, niter=20, seed=1234)
    ours.train(x)
    upstream = faiss.Kmeans(24, 10, niter=20, seed=1234, nredo=1)
    upstream.train(x)
    got_d, _ = ours.assign(x)
    ref_d, _ = upstream.assign(x)
    got_objective = float(got_d.sum())
    ref_objective = float(ref_d.sum())
    assert got_objective <= ref_objective * 1.15
    assert got_objective >= ref_objective * 0.85


def test_clone_is_independent(data):
    x, q = data
    original = mf.IndexFlatL2(24)
    original.add(x[:20])
    cloned = mf.clone_index(original)
    cloned.add(x[20:30])
    assert original.ntotal == 20
    assert cloned.ntotal == 30
    assert np.array_equal(original.reconstruct_n(), x[:20])
    assert np.array_equal(cloned.reconstruct_n(0, 20), x[:20])


def test_validation_errors(data):
    x, _ = data
    with pytest.raises(RuntimeError):
        mf.ProductQuantizer(24, 5)
    with pytest.raises(RuntimeError):
        mf.index_factory(24, "HNSW32")
    with pytest.raises(RuntimeError):
        mf.IndexPQ(24, 6, 4).add(x)
    with pytest.raises(ValueError):
        mf.IndexFlatL2(24).add(x[:, :10])
    ivf = mf.IndexIVFFlat(mf.IndexFlatL2(24), 24, 4)
    ivf.train(x)
    with pytest.raises(TypeError):
        ivf.add_with_ids(x[:2], np.array([1.0, 2.0]))
    with pytest.raises(RuntimeError):
        mf.IndexIVFFlat(mf.IndexFlatIP(24), 24, 4)
