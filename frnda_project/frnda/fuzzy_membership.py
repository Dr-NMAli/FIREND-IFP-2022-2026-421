

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fuzzy_knn_membership(X: np.ndarray, y: np.ndarray, K_f: int, m_f: float = 2.0,
                          classes: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:


    if m_f <= 1.0:
        raise ValueError("Fuzzy exponent m_f must be > 1.")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n = X.shape[0]

    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes)
    c = len(classes)
    class_to_col = {label: j for j, label in enumerate(classes)}
    y_col = np.array([class_to_col[label] for label in y])
    U_init = np.zeros((n, c))
    U_init[np.arange(n), y_col] = 1.0  # u_jk^init = 1[y_j = k]

    K_f = int(min(max(K_f, 1), n - 1))
    nbrs = NearestNeighbors(n_neighbors=K_f).fit(X)
    dist, idx = nbrs.kneighbors()  # no explicit X -> self-excluded, (n, K_f)

    d2 = dist ** 2
    zero_mask = d2 <= 0.0
    exponent = -1.0 / (m_f - 1.0)  # ||.||^{-2/(mf-1)} == (||.||^2)^{-1/(mf-1)}
    with np.errstate(divide="ignore"):
        w = np.where(zero_mask, 0.0, d2 ** exponent)  # (n, K_f) neighbor weights

    U_init_neighbors = U_init[idx]  # (n, K_f, c)
    numer = np.einsum("nk,nkc->nc", w, U_init_neighbors)  # (n, c)


    degenerate = zero_mask.any(axis=1)
    if degenerate.any():
        for i in np.nonzero(degenerate)[0]:
            coincident = idx[i][zero_mask[i]]
            numer[i] = U_init[coincident].mean(axis=0)

    denom = numer.sum(axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    U = numer / denom
    return U, classes
