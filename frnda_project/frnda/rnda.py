
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .choquet import AdditiveMeasure
from .graph import MultiScaleGraphs, graph_laplacian, symmetric_knn_support
from .optimize import frnda_optimize


def classical_scatter_matrices(X: np.ndarray, y: np.ndarray, ridge: float = 0.0
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:


    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape
    classes = np.unique(y)
    global_mean = X.mean(axis=0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for k in classes:
        Xk = X[y == k]
        mu_k = Xk.mean(axis=0)
        Xc = Xk - mu_k
        Sw += Xc.T @ Xc
        diff = (mu_k - global_mean).reshape(-1, 1)
        Sb += Xk.shape[0] * (diff @ diff.T)
    Sw = 0.5 * (Sw + Sw.T)
    Sb = 0.5 * (Sb + Sb.T)
    if ridge > 0:
        Sw = Sw + ridge * np.eye(d)
    return Sw, Sb, classes


def build_crisp_graphs(X: np.ndarray, scales: Sequence[int]) -> MultiScaleGraphs:


    X = np.asarray(X, dtype=float)
    scales = tuple(sorted(int(k) for k in scales))
    out = MultiScaleGraphs(scales=scales, sigma=None, kappa=0)
    for k in scales:
        A = symmetric_knn_support(X, k)
        L = graph_laplacian(A)
        B = X.T @ (L @ X)
        B = 0.5 * (B + B.T)
        out.A.append(A)
        out.L.append(L)
        out.B.append(B)
    return out


class RNDA(BaseEstimator, TransformerMixin):


    def __init__(self, n_components: int = 4, scales=(3, 5, 7, 9, 11), lam: float = 0.0,
                 gamma: float = 1e-2, ridge: float = 1e-6,
                 scale_weights: Optional[Sequence[float]] = None,
                 max_iter: int = 50, tol: float = 1e-6):
        self.n_components = n_components
        self.scales = scales
        self.lam = lam
        self.gamma = gamma
        self.ridge = ridge
        self.scale_weights = scale_weights
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RNDA":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, d = X.shape
        p = min(self.n_components, d)

        Sw, Sb, classes = classical_scatter_matrices(X, y, ridge=self.ridge)
        graphs = build_crisp_graphs(X, self.scales)
        M = len(graphs.scales)
        weights = (np.ones(M) / M if self.scale_weights is None
                   else np.asarray(self.scale_weights, dtype=float))
        measure = AdditiveMeasure(weights)

        result = frnda_optimize(Sw, Sb, graphs.B, measure, p=p,
                                 lam=self.lam, gamma=self.gamma, eps=self.ridge,
                                 max_iter=self.max_iter, tol=self.tol)
        self.W_ = result.W
        self.classes_ = classes
        self.Sw_ = Sw
        self.Sb_ = Sb
        self.graphs_ = graphs
        self.measure_ = measure
        self.history_ = result.history
        self.n_iter_ = result.n_iter
        self.converged_ = result.converged
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "W_")
        return np.asarray(X, dtype=float) @ self.W_

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.fit(X, y).transform(X)
