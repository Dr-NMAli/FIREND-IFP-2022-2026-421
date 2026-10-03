

import numpy as np
import pytest

from frnda import utils, graph


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ------------------------------------------------------------------ utils --
def test_pairwise_sq_dists_matches_brute_force(rng):
    X = rng.normal(size=(6, 3))
    Y = rng.normal(size=(5, 3))
    D2 = utils.pairwise_sq_dists(X, Y)
    D2_bf = np.array([[np.sum((X[i] - Y[j]) ** 2) for j in range(5)] for i in range(6)])
    assert np.allclose(D2, D2_bf)


def test_l21_norm_and_smoothed(rng):
    W = rng.normal(size=(6, 2))
    ref = sum(np.linalg.norm(W[i]) for i in range(6))
    assert utils.l21_norm(W) == pytest.approx(ref)
    # smoothed version -> raw version as eps -> 0
    assert utils.l21_norm_smoothed(W, 1e-12) == pytest.approx(ref, abs=1e-4)


def test_orthonormalize(rng):
    W = rng.normal(size=(7, 3))
    Q = utils.orthonormalize(W)
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    assert np.linalg.matrix_rank(np.hstack([Q, W])) == 3


def test_top_eigh_picks_largest(rng):
    M = rng.normal(size=(6, 6))
    M = M + M.T
    vals, vecs = utils.top_eigh(M, 4)
    all_vals = np.linalg.eigvalsh(M)
    assert np.allclose(sorted(vals), sorted(all_vals)[-4:])
    assert np.allclose(vecs.T @ vecs, np.eye(4), atol=1e-8)


def test_generalized_top_eigh_is_orthonormal(rng):
    A = rng.normal(size=(5, 5))
    A = A + A.T
    B = rng.normal(size=(5, 5))
    B = B @ B.T + 5 * np.eye(5)
    _, W = utils.generalized_top_eigh(A, B, 2)
    assert np.allclose(W.T @ W, np.eye(2), atol=1e-8)


# ------------------------------------------------------------------ graph --
def test_symmetric_knn_support_matches_brute_force(rng):
    X = rng.normal(size=(30, 4))
    k = 3
    n = X.shape[0]
    D2 = utils.pairwise_sq_dists(X)
    np.fill_diagonal(D2, np.inf)
    fwd_bf = np.zeros((n, n), dtype=bool)
    for i in range(n):
        fwd_bf[i, np.argsort(D2[i])[:k]] = True
    support_bf = fwd_bf | fwd_bf.T

    support = graph.symmetric_knn_support(X, k).toarray().astype(bool)
    assert np.array_equal(support, support_bf)


def test_local_bandwidth_matches_brute_force(rng):
    X = rng.normal(size=(30, 4))
    sigma = graph.local_bandwidth(X, kappa=5)
    D2 = utils.pairwise_sq_dists(X)
    np.fill_diagonal(D2, np.inf)
    sigma_bf = np.sort(np.sqrt(D2), axis=1)[:, 4]
    assert np.allclose(sigma, sigma_bf, atol=1e-8)


def test_fuzzy_rnn_affinity_symmetric_and_bounded(rng):
    X = rng.normal(size=(25, 4))
    sigma = graph.local_bandwidth(X, kappa=5)
    A = graph.fuzzy_rnn_affinity(X, 5, sigma).toarray()
    assert np.allclose(A, A.T)
    nz = A[A > 0]
    assert np.all(nz > 0) and np.all(nz <= 1.0)


def test_graph_laplacian_is_psd_with_zero_row_sums(rng):
    X = rng.normal(size=(25, 4))
    sigma = graph.local_bandwidth(X, kappa=5)
    A = graph.fuzzy_rnn_affinity(X, 5, sigma)
    L = graph.graph_laplacian(A).toarray()
    assert np.linalg.eigvalsh(L).min() > -1e-8
    assert np.allclose(L.sum(axis=1), 0, atol=1e-10)


def test_multi_scale_graphs_B_matrices_are_psd(rng):
    X = rng.normal(size=(30, 5))
    graphs = graph.build_multi_scale_graphs(X, scales=(3, 5, 7, 9, 11))
    assert graphs.kappa == 7  # median of (3,5,7,9,11)
    for B in graphs.B:
        assert np.linalg.eigvalsh(B).min() > -1e-6


def test_k_occurrence_counts_sum_to_nk(rng):
    X = rng.normal(size=(30, 4))
    counts = graph.k_occurrence_counts(X, k=5)
    assert counts.sum() == pytest.approx(30 * 5)
