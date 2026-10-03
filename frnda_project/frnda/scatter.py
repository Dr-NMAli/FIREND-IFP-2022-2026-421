

from __future__ import annotations

from typing import Tuple

import numpy as np


def fuzzy_class_means(X: np.ndarray, U: np.ndarray, m_f: float
                       ) -> Tuple[np.ndarray, np.ndarray]:


    Um = U ** m_f  # (n, c)
    weights = Um.sum(axis=0)  # (c,)
    safe_weights = np.where(weights > 0, weights, 1.0)
    means = (Um.T @ X) / safe_weights[:, None]  # (c, d)
    return means, weights


def fuzzy_scatter_matrices(X: np.ndarray, U: np.ndarray, m_f: float,
                            ridge: float = 0.0
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


    n, d = X.shape
    Um = U ** m_f
    s = Um.sum(axis=1)   # (n,)  sum_k u_ik^{mf}
    w = Um.sum(axis=0)   # (c,)  sum_i u_ik^{mf}
    safe_w = np.where(w > 0, w, 1.0)

    global_mean = X.mean(axis=0)
    Xc = X - global_mean
    means_c = (Um.T @ Xc) / safe_w[:, None]  # (c, d) == mu_k^f - x_bar

    Sw = (Xc * s[:, None]).T @ Xc - (means_c * w[:, None]).T @ means_c
    Sb = (means_c * w[:, None]).T @ means_c

    Sw = 0.5 * (Sw + Sw.T)
    Sb = 0.5 * (Sb + Sb.T)
    if ridge > 0:
        Sw = Sw + ridge * np.eye(d)

    means = means_c + global_mean
    return Sw, Sb, means, global_mean
