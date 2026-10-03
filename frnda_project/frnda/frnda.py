
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .choquet import AdditiveMeasure, FuzzyMeasure, TwoAdditiveMeasure
from .fuzzy_membership import fuzzy_knn_membership
from .graph import build_multi_scale_graphs
from .objective import trace_ratio_objective
from .optimize import frnda_optimize
from .scatter import fuzzy_scatter_matrices


class FRNDA(BaseEstimator, TransformerMixin):



    def __init__(self, n_components: int = 4, K_f: int = 5, m_f: float = 2.0,
                 scales=(3, 5, 7, 9, 11), lam: float = 1e-2, gamma: float = 1e-2,
                 ridge: float = 1e-6, measure: Union[str, FuzzyMeasure] = "two-additive",
                 mu_pairwise_scale: float = 0.5, max_iter: int = 50, tol: float = 1e-6,
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.K_f = K_f
        self.m_f = m_f
        self.scales = scales
        self.lam = lam
        self.gamma = gamma
        self.ridge = ridge
        self.measure = measure
        self.mu_pairwise_scale = mu_pairwise_scale
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # -- internals -----------------------------------------------------
    def _resolve_measure(self, M: int) -> FuzzyMeasure:
        if isinstance(self.measure, FuzzyMeasure):
            if self.measure.n_scales != M:
                raise ValueError("Provided fuzzy measure is defined over "
                                  f"{self.measure.n_scales} scales but {M} "
                                  "scales were requested via `scales`.")
            return self.measure
        if self.measure == "additive":
            return AdditiveMeasure.uniform(M)
        if self.measure == "two-additive":
            rng = np.random.default_rng(self.random_state)
            return TwoAdditiveMeasure.random_supermodular(
                M, rng=rng, pairwise_scale=self.mu_pairwise_scale)
        raise ValueError(f"Unknown measure spec: {self.measure!r}")

    # -- public API ------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "FRNDA":

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        p = min(self.n_components, d)

        U, classes = fuzzy_knn_membership(X, y, self.K_f, self.m_f)             
        Sw, Sb, means, gmean = fuzzy_scatter_matrices(X, U, self.m_f,
                                                        ridge=self.ridge)        
        graphs = build_multi_scale_graphs(X, self.scales)                        
        measure = self._resolve_measure(len(graphs.scales))                      

        result = frnda_optimize(Sw, Sb, graphs.B, measure, p=p,
                                 lam=self.lam, gamma=self.gamma, eps=self.ridge,
                                 max_iter=self.max_iter, tol=self.tol)            

        self.W_ = result.W
        self.U_ = U
        self.classes_ = classes
        self.Sw_ = Sw
        self.Sb_ = Sb
        self.class_means_ = means
        self.global_mean_ = gmean
        self.graphs_ = graphs
        self.measure_ = measure
        self.history_ = result.history
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        self.n_features_in_ = d
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
 
		check_is_fitted(self, "W_")
        X = np.asarray(X, dtype=float)
        return X @ self.W_

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def objective(self, W: Optional[np.ndarray] = None) -> float:
        
		check_is_fitted(self, "W_")
        W = self.W_ if W is None else W
        return trace_ratio_objective(W, self.Sw_, self.Sb_, self.graphs_.B,
                                      self.measure_, self.lam, self.gamma, self.ridge)
