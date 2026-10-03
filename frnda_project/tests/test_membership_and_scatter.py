
import numpy as np
import pytest

from frnda.fuzzy_membership import fuzzy_knn_membership
from frnda.scatter import fuzzy_scatter_matrices


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(1)
    n, d, c = 40, 5, 3
    X = rng.normal(size=(n, d))
    y = rng.integers(0, c, size=n)
    return X, y, rng


# ------------------------------------------------------------- membership --
def test_membership_is_a_probability_simplex(toy_data):
    X, y, _ = toy_data
    U, classes = fuzzy_knn_membership(X, y, K_f=5, m_f=2.0)
    assert U.shape == (40, 3)
    assert np.all(U >= -1e-12) and np.all(U <= 1 + 1e-12)
    assert np.allclose(U.sum(axis=1), 1.0, atol=1e-10)


def test_membership_matches_brute_force_row(toy_data):
    X, y, _ = toy_data
    K_f, m_f = 5, 2.0
    U, classes = fuzzy_knn_membership(X, y, K_f, m_f)
    class_to_col = {lab: j for j, lab in enumerate(classes)}

    i = 7
    D2 = np.sum((X - X[i]) ** 2, axis=1)
    D2[i] = np.inf
    nn = np.argsort(D2)[:K_f]
    num = np.zeros(3)
    for j in nn:
        num[class_to_col[y[j]]] += D2[j] ** (-1.0 / (m_f - 1.0))
    assert np.allclose(num / num.sum(), U[i], atol=1e-10)


def test_membership_homogeneous_neighborhood_is_near_crisp(toy_data):
    X, y, rng = toy_data
    # an isolated cluster of same-class points -> membership -> 1 for that class
    extra = 100 + rng.normal(scale=0.01, size=(6, X.shape[1]))
    X2 = np.vstack([X, extra])
    y2 = np.concatenate([y, np.zeros(6, dtype=int)])
    U, classes = fuzzy_knn_membership(X2, y2, K_f=5, m_f=2.0)
    assert U[len(X):, list(classes).index(0)].min() > 0.99


def test_membership_handles_duplicate_points_without_nan(toy_data):
    X, y, _ = toy_data
    Xd = np.vstack([X, X[3]])
    yd = np.concatenate([y, [y[3]]])
    U, _ = fuzzy_knn_membership(Xd, yd, K_f=5, m_f=2.0)
    assert not np.any(np.isnan(U))


def test_membership_rejects_mf_le_1(toy_data):
    X, y, _ = toy_data
    with pytest.raises(ValueError):
        fuzzy_knn_membership(X, y, K_f=5, m_f=1.0)


# ----------------------------------------------------------------- scatter --
def test_scatter_matches_brute_force_triple_loop(rng=np.random.default_rng(2)):
    n, d, c = 25, 6, 4
    X = rng.normal(size=(n, d)) * 3 + 10  # nonzero mean, stresses numerical stability
    U = rng.dirichlet(np.ones(c), size=n)
    m_f = 2.0

    Sw, Sb, means, gmean = fuzzy_scatter_matrices(X, U, m_f)

    Um = U ** m_f
    means_bf = (Um.T @ X) / Um.sum(axis=0)[:, None]
    w_bf = Um.sum(axis=0)
    assert np.allclose(means, means_bf, atol=1e-8)

    Sw_bf = np.zeros((d, d))
    for i in range(n):
        for k in range(c):
            diff = (X[i] - means_bf[k]).reshape(-1, 1)
            Sw_bf += Um[i, k] * (diff @ diff.T)
    Sb_bf = np.zeros((d, d))
    gm = X.mean(axis=0)
    for k in range(c):
        diff = (means_bf[k] - gm).reshape(-1, 1)
        Sb_bf += w_bf[k] * (diff @ diff.T)

    assert np.allclose(Sw, Sw_bf, atol=1e-6)
    assert np.allclose(Sb, Sb_bf, atol=1e-6)


def test_proposition_psd_of_fuzzy_scatter_matrices():
    rng = np.random.default_rng(3)
    n, d, c = 20, 5, 3
    for _ in range(20):
        U = rng.dirichlet(np.ones(c), size=n)
        m_f = rng.uniform(1.0, 4.0)
        X = rng.normal(size=(n, d))
        Sw, Sb, _, _ = fuzzy_scatter_matrices(X, U, m_f)
        assert np.linalg.eigvalsh(Sw).min() > -1e-8
        assert np.linalg.eigvalsh(Sb).min() > -1e-8


def test_proposition_reduction_crisp_membership_gives_classical_lda_scatter():
    rng = np.random.default_rng(4)
    n, d, c = 25, 6, 4
    X = rng.normal(size=(n, d)) * 3 + 10
    y = rng.integers(0, c, size=n)
    U_crisp = np.eye(c)[y]

    Sw_f, Sb_f, _, _ = fuzzy_scatter_matrices(X, U_crisp, m_f=2.0)

    gm = X.mean(axis=0)
    Sw_lda = np.zeros((d, d))
    Sb_lda = np.zeros((d, d))
    for k in range(c):
        Xk = X[y == k]
        mu_k = Xk.mean(axis=0)
        Xc = Xk - mu_k
        Sw_lda += Xc.T @ Xc
        diff = (mu_k - gm).reshape(-1, 1)
        Sb_lda += Xk.shape[0] * (diff @ diff.T)

    assert np.allclose(Sw_f, Sw_lda, atol=1e-8)
    assert np.allclose(Sb_f, Sb_lda, atol=1e-8)


def test_ridge_regularization_applied():
    rng = np.random.default_rng(5)
    n, d, c = 20, 4, 3
    X = rng.normal(size=(n, d))
    U = rng.dirichlet(np.ones(c), size=n)
    Sw, _, _, _ = fuzzy_scatter_matrices(X, U, 2.0)
    Sw_r, _, _, _ = fuzzy_scatter_matrices(X, U, 2.0, ridge=0.1)
    assert np.allclose(Sw_r, Sw + 0.1 * np.eye(d))
