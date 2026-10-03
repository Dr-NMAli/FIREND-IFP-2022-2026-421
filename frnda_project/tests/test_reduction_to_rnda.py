

import numpy as np
from scipy.linalg import subspace_angles

from frnda.choquet import AdditiveMeasure
from frnda.graph import MultiScaleGraphs, fuzzy_rnn_affinity, graph_laplacian, symmetric_knn_support
from frnda.optimize import frnda_optimize
from frnda.rnda import RNDA, build_crisp_graphs, classical_scatter_matrices
from frnda.scatter import fuzzy_scatter_matrices


def _problem(seed=11, n=60, d=8, c=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, c, size=n)
    return X, y


def test_condition_i_crisp_membership_gives_classical_scatter():
    X, y = _problem()
    c = len(np.unique(y))
    U_crisp = np.eye(c)[y]
    Sw_f, Sb_f, _, _ = fuzzy_scatter_matrices(X, U_crisp, m_f=2.0, ridge=1e-6)
    Sw_c, Sb_c, _ = classical_scatter_matrices(X, y, ridge=1e-6)
    assert np.allclose(Sw_f, Sw_c, atol=1e-8)
    assert np.allclose(Sb_f, Sb_c, atol=1e-8)


def test_condition_ii_huge_sigma_gives_binary_graph():
    X, y = _problem()
    scales = (3, 5, 7)
    huge_sigma = np.full(X.shape[0], 1e6)
    for k in scales:
        A_fuzzy = fuzzy_rnn_affinity(X, k, huge_sigma).toarray()
        A_binary = symmetric_knn_support(X, k).toarray()
        assert np.allclose(A_fuzzy, A_binary, atol=1e-6)


def test_full_reduction_end_to_end_matches_independent_rnda():
    X, y = _problem()
    scales = (3, 5, 7)
    c = len(np.unique(y))
    p = 3


    U_crisp = np.eye(c)[y]
    Sw_f, Sb_f, _, _ = fuzzy_scatter_matrices(X, U_crisp, m_f=2.0, ridge=1e-6)

    huge_sigma = np.full(X.shape[0], 1e6)
    graphs_fuzzy_limit = MultiScaleGraphs(scales=scales)
    for k in scales:
        A = fuzzy_rnn_affinity(X, k, huge_sigma)
        L = graph_laplacian(A)
        B = X.T @ (L @ X)
        graphs_fuzzy_limit.B.append(0.5 * (B + B.T))

    measure = AdditiveMeasure.uniform(len(scales))
    res_fuzzy_limit = frnda_optimize(Sw_f, Sb_f, graphs_fuzzy_limit.B, measure, p=p,
                                      lam=0.0, gamma=1e-2, eps=1e-6, max_iter=50, tol=1e-10)


    Sw_c, Sb_c, _ = classical_scatter_matrices(X, y, ridge=1e-6)
    graphs_rnda = build_crisp_graphs(X, scales)
    res_rnda = frnda_optimize(Sw_c, Sb_c, graphs_rnda.B, measure, p=p,
                               lam=0.0, gamma=1e-2, eps=1e-6, max_iter=50, tol=1e-10)

    assert abs(res_fuzzy_limit.history[-1] - res_rnda.history[-1]) < 1e-4

    angles = subspace_angles(res_fuzzy_limit.W, res_rnda.W)
    assert np.allclose(angles, 0, atol=1e-3)


def test_rnda_estimator_runs_end_to_end():
    X, y = _problem(n=80, d=10, c=3)
    model = RNDA(n_components=3, scales=(3, 5, 7))
    Z = model.fit_transform(X, y)
    assert Z.shape == (80, 3)
    assert np.diff(model.history_).min() > -1e-7
    assert np.allclose(model.W_.T @ model.W_, np.eye(3), atol=1e-6)
