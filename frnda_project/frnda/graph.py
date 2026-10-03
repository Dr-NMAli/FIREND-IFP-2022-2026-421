
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors, kneighbors_graph


def local_bandwidth(X: np.ndarray, kappa: int, metric: str = "euclidean") -> np.ndarray:

    n = X.shape[0]
    kappa = int(min(max(kappa, 1), n - 1))
    nbrs = NearestNeighbors(n_neighbors=kappa, metric=metric).fit(X)
    dist, _ = nbrs.kneighbors()  # no explicit X -> self-excluded by sklearn
    sigma = dist[:, -1].copy()

    positive = sigma[sigma > 0]
    floor = positive.min() * 1e-3 if positive.size else 1e-12
    sigma[sigma <= 0] = floor
    return sigma


def symmetric_knn_support(X: np.ndarray, k: int, metric: str = "euclidean") -> sp.csr_matrix:


    n = X.shape[0]
    k = int(min(max(k, 1), n - 1))
    fwd = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                            include_self=False, metric=metric)
    support = fwd.maximum(fwd.T).tocsr()  # elementwise OR of {0,1} entries
    support.data[:] = 1.0
    return support


def fuzzy_rnn_affinity(X: np.ndarray, k: int, sigma: np.ndarray,
                        metric: str = "euclidean") -> sp.csr_matrix:



    support = symmetric_knn_support(X, k, metric=metric).tocoo()
    rows, cols = support.row, support.col
    diff = X[rows] - X[cols]
    d2 = np.einsum("ij,ij->i", diff, diff)
    weights = np.exp(-d2 / (2.0 * sigma[rows] * sigma[cols]))
    A = sp.coo_matrix((weights, (rows, cols)), shape=(X.shape[0], X.shape[0]))
    return A.tocsr()


def graph_laplacian(A: sp.spmatrix) -> sp.csr_matrix:

    degrees = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(degrees)
    return (D - A).tocsr()


@dataclass
class MultiScaleGraphs:


    scales: tuple
    A: List = field(default_factory=list)
    L: List = field(default_factory=list)
    B: List[np.ndarray] = field(default_factory=list)
    sigma: Optional[np.ndarray] = None
    kappa: int = 0


def build_multi_scale_graphs(X: np.ndarray, scales: Sequence[int],
                              metric: str = "euclidean") -> MultiScaleGraphs:


    X = np.asarray(X, dtype=float)
    scales = tuple(sorted(int(k) for k in scales))
    kappa = int(np.median(scales))
    sigma = local_bandwidth(X, kappa, metric=metric)

    out = MultiScaleGraphs(scales=scales, sigma=sigma, kappa=kappa)
    for k in scales:
        A = fuzzy_rnn_affinity(X, k, sigma, metric=metric)
        L = graph_laplacian(A)
        B = X.T @ (L @ X)  # X^T L_m^f X  (== paper's X L_m^f X^T under X: d x n)
        B = 0.5 * (B + B.T)
        out.A.append(A)
        out.L.append(L)
        out.B.append(B)
    return out


def k_occurrence_counts(X: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:

    fwd = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                            include_self=False, metric=metric)
    return np.asarray(fwd.sum(axis=0)).ravel()
