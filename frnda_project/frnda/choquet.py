"""
Fuzzy measures and the discrete Choquet integral.
"""
from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np


class FuzzyMeasure:
    """Abstract base class for a capacity :math:`g: 2^{\\mathcal M} \\to
    [0,1]` with :math:`g(\\emptyset)=0`, :math:`g(\\mathcal M)=1`, and
    monotonicity ( "Fuzzy measure"). Subclasses implement
    ``value(subset)``.
    """

    n_scales: int

    def value(self, subset: frozenset) -> float:
        raise NotImplementedError

    def is_monotone(self, n_checks: int = 300, rng: "np.random.Generator | None" = None) -> bool:
        """Randomized monotonicity check g(A) <= g(B) for sampled A subset B
        (a sanity/debug utility, not required for normal use)."""
        rng = rng or np.random.default_rng(0)
        M = self.n_scales
        for _ in range(n_checks):
            A = frozenset(i for i in range(M) if rng.random() < 0.5)
            B = A | frozenset(i for i in range(M) if i not in A and rng.random() < 0.5)
            if self.value(A) > self.value(B) + 1e-9:
                return False
        return True


class AdditiveMeasure(FuzzyMeasure):

    def __init__(self, weights: Iterable[float]):
        w = np.asarray(list(weights), dtype=float)
        if np.any(w < 0):
            raise ValueError("Additive-measure weights must be nonnegative.")
        total = w.sum()
        if total <= 0:
            raise ValueError("Additive-measure weights must sum to a positive value.")
        self.weights = w / total
        self.n_scales = len(w)

    @classmethod
    def uniform(cls, M: int) -> "AdditiveMeasure":
        return cls(np.ones(M) / M)

    def value(self, subset: frozenset) -> float:
        if not subset:
            return 0.0
        return float(self.weights[list(subset)].sum())


class TwoAdditiveMeasure(FuzzyMeasure):

    def __init__(self, mu_singleton: Iterable[float], mu_pairwise: np.ndarray):
        mu_s = np.asarray(list(mu_singleton), dtype=float)
        mu_p = np.asarray(mu_pairwise, dtype=float)
        M = len(mu_s)
        if mu_p.shape != (M, M):
            raise ValueError("mu_pairwise must be an (M, M) array.")
        if np.any(mu_s <= 0):
            raise ValueError("Singleton importances mu_m must be strictly positive.")
        upper = np.triu(mu_p, k=1)
        if np.any(upper < -1e-12):
            raise ValueError("Pairwise interactions mu_{m,m'} must be nonnegative "
                              "to guarantee supermodularity (Eq. 2additive).")
        upper = np.clip(upper, 0.0, None)
        mu_p_sym = upper + upper.T  # symmetric storage, zero diagonal

        total = mu_s.sum() + upper.sum()
        if total <= 0:
            raise ValueError("Measure mass must be positive.")
        self.mu_singleton = mu_s / total
        self.mu_pairwise = mu_p_sym / total
        self.n_scales = M

    @classmethod
    def random_supermodular(cls, M: int, rng: "np.random.Generator | None" = None,
                             pairwise_scale: float = 0.5) -> "TwoAdditiveMeasure":

        rng = rng or np.random.default_rng()
        mu_s = rng.uniform(0.5, 1.5, size=M)
        upper = rng.uniform(0.0, pairwise_scale, size=(M, M))
        upper = np.triu(upper, k=1)
        mu_p = upper + upper.T
        return cls(mu_s, mu_p)

    def value(self, subset: frozenset) -> float:
        if not subset:
            return 0.0
        idx = list(subset)
        val = self.mu_singleton[idx].sum()
        if len(idx) >= 2:
            sub_p = self.mu_pairwise[np.ix_(idx, idx)]
            val += np.triu(sub_p, k=1).sum()
        return float(val)

    def shapley_values(self) -> np.ndarray:

        return self.mu_singleton + 0.5 * self.mu_pairwise.sum(axis=1)

    def interaction_indices(self) -> np.ndarray:

        return self.mu_pairwise.copy()


def choquet_integral(r: np.ndarray, measure: FuzzyMeasure, return_coeffs: bool = False
                      ) -> Union[float, Tuple[float, np.ndarray]]:
    r"""
    Parameters
    ----------
    r : (M,) array_like, nonnegative.
    measure : FuzzyMeasure
    return_coeffs : bool, default False
        If True, also return the (M,) coefficient vector c (in the
        original, unsorted coordinate order).

    Returns
    -------
    value : float
    coeffs : (M,) ndarray, only if return_coeffs=True.
    """
    r = np.asarray(r, dtype=float)
    M = r.shape[0]
    if M != measure.n_scales:
        raise ValueError(f"r has {M} entries but the measure is defined on "
                          f"{measure.n_scales} scales.")
    if M == 0:
        return (0.0, np.zeros(0)) if return_coeffs else 0.0

    order = np.argsort(r, kind="stable")  # ascending: order[0] = index of r_(1)
    g_tail = np.empty(M + 1)
    for pos in range(M):
        subset = frozenset(order[pos:].tolist())  # A_(m), m = pos + 1
        g_tail[pos] = measure.value(subset)
    g_tail[M] = 0.0  # g(A_(M+1)) := g(empty) = 0

    coeffs_sorted = g_tail[:-1] - g_tail[1:]  # c_(m) = g(A_(m)) - g(A_(m+1))
    coeffs = np.empty(M)
    coeffs[order] = coeffs_sorted

    value = float(np.dot(r, coeffs))

    if return_coeffs:
        return value, coeffs
    return value
