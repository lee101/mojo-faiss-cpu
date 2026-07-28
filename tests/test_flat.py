import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

import mojofaiss as mf


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(42)
    xb = rng.normal(size=(257, 13)).astype(np.float32)
    xq = rng.normal(size=(19, 13)).astype(np.float32)
    return xb, xq


@pytest.mark.parametrize(
    ("ours_cls", "upstream_cls"),
    [(mf.IndexFlatL2, faiss.IndexFlatL2), (mf.IndexFlatIP, faiss.IndexFlatIP)],
)
def test_flat_search_parity(vectors, ours_cls, upstream_cls):
    xb, xq = vectors
    ours, upstream = ours_cls(13), upstream_cls(13)
    ours.add(xb)
    upstream.add(xb)
    got_d, got_i = ours.search(xq, 17)
    ref_d, ref_i = upstream.search(xq, 17)
    assert got_d.dtype == np.float32
    assert got_i.dtype == np.int64
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    ("ours_cls", "upstream_cls"),
    [(mf.IndexFlatL2, faiss.IndexFlatL2), (mf.IndexFlatIP, faiss.IndexFlatIP)],
)
def test_flat_padding_parity(vectors, ours_cls, upstream_cls):
    xb, xq = vectors
    ours, upstream = ours_cls(13), upstream_cls(13)
    ours.add(xb[:3])
    upstream.add(xb[:3])
    got_d, got_i = ours.search(xq[:2], 7)
    ref_d, ref_i = upstream.search(xq[:2], 7)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d[:, :3], ref_d[:, :3], rtol=2e-6, atol=2e-6)
    assert np.array_equal(got_d[:, 3:], ref_d[:, 3:])


@pytest.mark.parametrize(
    ("ours_cls", "upstream_cls", "threshold"),
    [
        (mf.IndexFlatL2, faiss.IndexFlatL2, 22.0),
        (mf.IndexFlatIP, faiss.IndexFlatIP, 2.0),
    ],
)
def test_range_search_parity(vectors, ours_cls, upstream_cls, threshold):
    xb, xq = vectors
    ours, upstream = ours_cls(13), upstream_cls(13)
    ours.add(xb)
    upstream.add(xb)
    got_lims, got_d, got_i = ours.range_search(xq[:5], threshold)
    ref_lims, ref_d, ref_i = upstream.range_search(xq[:5], threshold)
    assert np.array_equal(got_lims, ref_lims)
    for q in range(5):
        gs = slice(got_lims[q], got_lims[q + 1])
        rs = slice(ref_lims[q], ref_lims[q + 1])
        go = np.argsort(got_i[gs])
        ro = np.argsort(ref_i[rs])
        assert np.array_equal(got_i[gs][go], ref_i[rs][ro])
        assert np.allclose(got_d[gs][go], ref_d[rs][ro], rtol=2e-6, atol=2e-6)


def test_reconstruct_reset_and_distance_subset(vectors):
    xb, xq = vectors
    index = mf.IndexFlatL2(13)
    index.add(xb[:30])
    assert index.ntotal == 30
    assert np.array_equal(index.reconstruct(7), xb[7])
    labels = np.array([[0, 4, 8], [3, 2, 1]], dtype=np.int64)
    got = index.compute_distance_subset(xq[:2], labels)
    ref = np.stack(
        [np.sum((xb[row] - xq[q]) ** 2, axis=1) for q, row in enumerate(labels)]
    )
    assert np.allclose(got, ref)
    index.reset()
    assert index.ntotal == 0
    distances, ids = index.search(xq[:1], 2)
    assert np.all(ids == -1)
    assert np.all(distances == np.finfo(np.float32).max)


def test_assign_reconstruct_n_and_search_and_reconstruct(vectors):
    xb, xq = vectors
    index = mf.IndexFlatL2(13)
    index.add(xb[:30])
    distances, labels, reconstructed = index.search_and_reconstruct(xq[:3], 4)
    assert np.array_equal(index.assign(xq[:3], 4), labels)
    assert np.array_equal(index.reconstruct_n(5, 3), xb[5:8])
    assert np.array_equal(reconstructed, xb[labels])
    assert distances.shape == (3, 4)
    assert index.reconstruct_n(index.ntotal, 0).shape == (0, 13)


def test_normalize_l2_parity(vectors):
    xb, _ = vectors
    got = xb[:20].copy()
    ref = xb[:20].copy()
    mf.normalize_L2(got)
    faiss.normalize_L2(ref)
    assert np.allclose(got, ref, rtol=1e-6, atol=1e-7)


def test_knn_function_parity(vectors):
    xb, xq = vectors
    got_d, got_i = mf.knn(xq, xb, 9)
    ref_d, ref_i = faiss.knn(xq, xb, 9)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=2e-6)


def test_id_map_parity(vectors):
    xb, xq = vectors
    ids = np.arange(len(xb), dtype=np.int64) * 7 + 100
    ours = mf.IndexIDMap2(mf.IndexFlatL2(13))
    upstream = faiss.IndexIDMap2(faiss.IndexFlatL2(13))
    ours.add_with_ids(xb, ids)
    upstream.add_with_ids(xb, ids)
    got_d, got_i = ours.search(xq, 8)
    ref_d, ref_i = upstream.search(xq, 8)
    assert np.array_equal(got_i, ref_i)
    assert np.allclose(got_d, ref_d, rtol=2e-6, atol=2e-6)
    assert np.array_equal(ours.reconstruct(int(ids[4])), xb[4])
    ours.reset()
    assert ours.ntotal == 0


def test_boundary_validation_and_empty_queries(vectors):
    xb, _ = vectors
    index = mf.IndexFlatL2(13)
    index.add(xb[:5])
    distances, labels = index.search(np.empty((0, 13), dtype=np.float32), 3)
    assert distances.shape == (0, 3)
    assert labels.shape == (0, 3)
    with pytest.raises(TypeError):
        index.add(xb[:1].astype(np.float64))
    with pytest.raises(TypeError):
        index.search(xb[:1], 1.5)
    with pytest.raises(ValueError):
        index.compute_distance_subset(
            xb[:1], np.array([[-1]], dtype=np.int64)
        )
