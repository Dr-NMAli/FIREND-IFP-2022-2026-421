
from itertools import combinations, permutations

import numpy as np
import pytest

from frnda.choquet import AdditiveMeasure, TwoAdditiveMeasure, choquet_integral


M = 5


def _brute_force_choquet(r, measure):
    order = np.argsort(r)
    r_sorted = r[order]
    total, r_prev = 0.0, 0.0
    for m in range(1, M + 1):
        subset = frozenset(order[m - 1:].tolist())
        total += (r_sorted[m - 1] - r_prev) * measure.value(subset)
        r_prev = r_sorted[m - 1]
    return total


@pytest.fixture
def rng():
    return np.random.default_rng(3)


def test_measure_boundary_conditions(rng):
    for _ in range(10):
        meas = TwoAdditiveMeasure.random_supermodular(M, rng=rng)
        assert meas.value(frozenset(range(M))) == pytest.approx(1.0)
        assert meas.value(frozenset()) == 0.0


def test_measure_is_monotone_and_supermodular(rng):
    all_subsets = [frozenset(c) for size in range(M + 1) for c in combinations(range(M), size)]
    for _ in range(5):
        meas = TwoAdditiveMeasure.random_supermodular(M, rng=rng)
        assert meas.is_monotone(n_checks=500, rng=rng)
        for A in all_subsets:
            for B in all_subsets:
                lhs = meas.value(A | B) + meas.value(A & B)
                rhs = meas.value(A) + meas.value(B)
                assert lhs >= rhs - 1e-9


def test_choquet_value_matches_definition_and_internality(rng):
    for _ in range(30):
        meas = TwoAdditiveMeasure.random_supermodular(M, rng=rng)
        r = rng.uniform(0, 5, size=M)
        val_bf = _brute_force_choquet(r, meas)
        val, coeffs = choquet_integral(r, meas, return_coeffs=True)
        assert val == pytest.approx(val_bf, abs=1e-9)
        assert r.min() - 1e-9 <= val <= r.max() + 1e-9  


def test_choquet_gradient_matches_finite_differences(rng):
    eps = 1e-6
    for _ in range(20):
        meas = TwoAdditiveMeasure.random_supermodular(M, rng=rng)
        r = rng.uniform(1, 5, size=M)  
        _, coeffs = choquet_integral(r, meas, return_coeffs=True)
        for m in range(M):
            r_plus, r_minus = r.copy(), r.copy()
            r_plus[m] += eps
            r_minus[m] -= eps
            num_grad = (choquet_integral(r_plus, meas) - choquet_integral(r_minus, meas)) / (2 * eps)
            assert num_grad == pytest.approx(coeffs[m], abs=1e-4)


def test_additive_measure_reduces_to_linear_averaging(rng):
   
	for _ in range(20):
        w = rng.dirichlet(np.ones(M))
        meas = AdditiveMeasure(w)
        r = rng.uniform(0, 10, size=M)
        val, coeffs = choquet_integral(r, meas, return_coeffs=True)
        assert np.allclose(coeffs, meas.weights, atol=1e-10)  
        assert val == pytest.approx(np.dot(r, meas.weights), abs=1e-9)


def test_comonotonic_minimality(rng):
    
	meas = TwoAdditiveMeasure.random_supermodular(M, rng=rng)
    r = rng.uniform(0, 5, size=M)
    true_val, _ = choquet_integral(r, meas, return_coeffs=True)

    worst = 0.0
    for perm in permutations(range(M)):
        g_tail = [meas.value(frozenset(perm[j:])) for j in range(M)] + [0.0]
        c_pi = np.zeros(M)
        for j in range(M):
            c_pi[perm[j]] = g_tail[j] - g_tail[j + 1]
        surrogate = float(np.dot(c_pi, r))
        worst = min(worst, surrogate - true_val)
        if np.array_equal(np.array(perm), np.argsort(r)):
            assert surrogate == pytest.approx(true_val, abs=1e-9)
    assert worst >= -1e-9


def test_uniform_additive_measure_classmethod():
    meas = AdditiveMeasure.uniform(4)
    assert np.allclose(meas.weights, 0.25)


def test_shapley_and_interaction_indices_shapes(rng):
    meas = TwoAdditiveMeasure.random_supermodular(M, rng=rng)
    assert meas.shapley_values().shape == (M,)
    assert meas.interaction_indices().shape == (M, M)
    assert np.allclose(np.diag(meas.interaction_indices()), 0.0)


def test_rejects_negative_pairwise_interactions():
    with pytest.raises(ValueError):
        TwoAdditiveMeasure(mu_singleton=[1, 1, 1], mu_pairwise=-np.ones((3, 3)))
