
import numpy as np
import pytest
from sklearn.datasets import make_classification

from frnda import FRNDA
from frnda.choquet import TwoAdditiveMeasure
from frnda.fuzzy_membership import fuzzy_knn_membership
from frnda.graph import build_multi_scale_graphs
from frnda.optimize import frnda_optimize
from frnda.scatter import fuzzy_scatter_matrices
from frnda.utils import orthonormalize


def _make_problem(n=150, d=20, n_classes=5, seed=7):
    X, y = make_classification(n_samples=n, n_features=d, n_informative=min(10, d),
                                n_classes=n_classes, n_clusters_per_class=1,
                                class_sep=0.8, random_state=seed)
    X = (X - X.mean(0)) / X.std(0)
    U, classes = fuzzy_knn_membership(X, y, K_f=5, m_f=2.0)
    Sw, Sb, _, _ = fuzzy_scatter_matrices(X, U, 2.0, ridge=1e-6)
    graphs = build_multi_scale_graphs(X, scales=(3, 5, 7, 9, 11))
    return Sw, Sb, graphs


def test_single_run_monotonic_and_orthonormal():
    Sw, Sb, graphs = _make_problem()
    measure = TwoAdditiveMeasure.random_supermodular(len(graphs.scales), rng=np.random.default_rng(42))
    p = 4
    result = frnda_optimize(Sw, Sb, graphs.B, measure, p=p, lam=1e-2, gamma=1e-2,
                             eps=1e-6, max_iter=60, tol=1e-9)
    diffs = np.diff(result.history)
    assert diffs.min() > -1e-8, "Theorem 'Monotonic convergence' violated"
    assert np.allclose(result.W.T @ result.W, np.eye(p), atol=1e-6)


def test_six_initializations_converge_to_common_value():

    Sw, Sb, graphs = _make_problem()
    measure = TwoAdditiveMeasure.random_supermodular(len(graphs.scales), rng=np.random.default_rng(123))
    d, p = Sw.shape[0], 4
    rng = np.random.default_rng(999)

    inits = [None] + [orthonormalize(rng.normal(size=(d, p))) for _ in range(5)]
    finals = []
    for W0 in inits:
        res = frnda_optimize(Sw, Sb, graphs.B, measure, p=p, lam=1e-2, gamma=1e-2,
                              eps=1e-6, max_iter=100, tol=1e-10, W_init=W0)
        assert np.diff(res.history).min() > -1e-7
        finals.append(res.history[-1])

    finals = np.array(finals)
    assert finals.max() - finals.min() < 1e-4


def test_frnda_estimator_end_to_end():
    X, y = make_classification(n_samples=120, n_features=15, n_informative=8,
                                n_classes=3, n_clusters_per_class=1, random_state=0)
    model = FRNDA(n_components=4, K_f=5, m_f=2.0, random_state=0)
    Z = model.fit_transform(X, y)
    assert Z.shape == (120, 4)
    assert np.diff(model.history_).min() > -1e-7
    assert np.allclose(model.W_.T @ model.W_, np.eye(4), atol=1e-6)
    # transform on new data uses the learned W_
    Z2 = model.transform(X[:10])
    assert np.allclose(Z2, X[:10] @ model.W_)


def test_frnda_rejects_scale_mismatched_measure():
    from frnda.choquet import AdditiveMeasure
    X, y = make_classification(n_samples=60, n_features=10, n_classes=2, random_state=0)
    bad_measure = AdditiveMeasure.uniform(3)  # doesn't match default 5 scales
    model = FRNDA(n_components=2, measure=bad_measure, random_state=0)
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_frnda_get_set_params_sklearn_compatible():
    model = FRNDA(n_components=3, K_f=7)
    params = model.get_params()
    assert params["n_components"] == 3 and params["K_f"] == 7
    model.set_params(K_f=9)
    assert model.K_f == 9
