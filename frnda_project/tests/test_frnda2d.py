
import numpy as np
import pytest

from frnda import FRNDA2D
from frnda.fuzzy_membership import fuzzy_knn_membership
from frnda.frnda2d import _mode_scatter, col_mode_matrices, graph_weighted_scatter, row_mode_matrices
from frnda.graph import build_multi_scale_graphs
from frnda.utils import orthonormalize


@pytest.fixture
def image_problem():
    rng = np.random.default_rng(21)
    n, h, w, c = 40, 6, 5, 3
    images = rng.normal(size=(n, h, w))
    y = rng.integers(0, c, size=n)
    scales = (3, 5, 7)
    flat = images.reshape(n, h * w)
    U_mem, classes = fuzzy_knn_membership(flat, y, K_f=5, m_f=2.0)
    graphs = build_multi_scale_graphs(flat, scales)
    return images, y, U_mem, graphs, rng


def test_mode_scatter_matches_brute_force(image_problem):
    images, y, U_mem, graphs, rng = image_problem
    n, h, w = images.shape
    c = U_mem.shape[1]
    m_f = 2.0
    p2 = 2
    V = orthonormalize(rng.normal(size=(w, p2)))

    Y = np.einsum("nhw,wp->nhp", images, V)
    Sw, Sb = _mode_scatter(U_mem, m_f, Y)

    Um = U_mem ** m_f
    Mk = np.zeros((c, h, w))
    wk = np.zeros(c)
    for k in range(c):
        num, den = np.zeros((h, w)), 0.0
        for i in range(n):
            num += Um[i, k] * images[i]
            den += Um[i, k]
        Mk[k], wk[k] = num / den, den
    Mbar = images.mean(axis=0)

    Sw_bf = np.zeros((h, h))
    Sb_bf = np.zeros((h, h))
    for i in range(n):
        for k in range(c):
            diff = (images[i] - Mk[k]) @ V
            Sw_bf += Um[i, k] * (diff @ diff.T)
    for k in range(c):
        diff = (Mk[k] - Mbar) @ V
        Sb_bf += wk[k] * (diff @ diff.T)

    assert np.allclose(Sw, Sw_bf, atol=1e-6)
    assert np.allclose(Sb, Sb_bf, atol=1e-6)


def test_graph_weighted_scatter_matches_brute_force(image_problem):
    images, y, U_mem, graphs, rng = image_problem
    n, h, w = images.shape
    V = orthonormalize(rng.normal(size=(w, 2)))
    Y = np.einsum("nhw,wp->nhp", images, V)

    A0 = graphs.A[0]
    B_computed = graph_weighted_scatter(A0, Y)

    A0d = A0.toarray()
    B_bf = np.zeros((h, h))
    for i in range(n):
        for j in range(n):
            if A0d[i, j] != 0:
                diff = Y[i] - Y[j]
                B_bf += A0d[i, j] * (diff @ diff.T)
    assert np.allclose(B_computed, B_bf, atol=1e-6)


def test_lemma_mode_consistency(image_problem):
    images, y, U_mem, graphs, rng = image_problem
    n, h, w = images.shape
    m_f = 2.0
    p1, p2 = 3, 2
    U = orthonormalize(rng.normal(size=(h, p1)))
    V = orthonormalize(rng.normal(size=(w, p2)))

    Sw_r, Sb_r, B_r = row_mode_matrices(images, U_mem, m_f, V, graphs.A, ridge=1e-6)
    Sw_c, Sb_c, B_c = col_mode_matrices(images, U_mem, m_f, U, graphs.A, ridge=1e-6)

    assert np.trace(U.T @ Sb_r @ U) == pytest.approx(np.trace(V.T @ Sb_c @ V), abs=1e-6)
    Sw_r_noridge = Sw_r - 1e-6 * np.eye(h)
    Sw_c_noridge = Sw_c - 1e-6 * np.eye(w)
    assert np.trace(U.T @ Sw_r_noridge @ U) == pytest.approx(np.trace(V.T @ Sw_c_noridge @ V), abs=1e-6)
    for Bm_r, Bm_c in zip(B_r, B_c):
        assert np.trace(U.T @ Bm_r @ U) == pytest.approx(np.trace(V.T @ Bm_c @ V), abs=1e-6)


def test_frnda2d_end_to_end_convergence(image_problem):
    images, y, *_ = image_problem
    model = FRNDA2D(p1_components=3, p2_components=2, K_f=5, m_f=2.0,
                     scales=(3, 5, 7), lam=1e-3, gamma=1e-2,
                     outer_max_iter=15, random_state=0)
    model.fit(images, y)

    diffs = np.diff(model.history_)
    if len(diffs):
        assert diffs.min() > -1e-6, 'Corollary "2DFRNDA convergence" violated'
    assert np.allclose(model.U_.T @ model.U_, np.eye(3), atol=1e-6)
    assert np.allclose(model.V_.T @ model.V_, np.eye(2), atol=1e-6)

    Y = model.transform(images[:5])
    assert Y.shape == (5, 3, 2)
    Y_single = model.transform(images[0])
    assert Y_single.shape == (3, 2)
    assert np.allclose(Y[0], Y_single)


def test_frnda2d_rejects_non_3d_input():
    model = FRNDA2D()
    with pytest.raises(ValueError):
        model.fit(np.zeros((10, 20)), np.zeros(10, dtype=int))
