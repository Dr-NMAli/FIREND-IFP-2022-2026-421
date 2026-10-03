
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .choquet import AdditiveMeasure, FuzzyMeasure, TwoAdditiveMeasure
from .fuzzy_membership import fuzzy_knn_membership
from .graph import build_multi_scale_graphs
from .objective import trace_ratio_objective
from .optimize import frnda_optimize
from .utils import orthonormalize


def graph_weighted_scatter(A: sp.spmatrix, Y: np.ndarray) -> np.ndarray:

    A = A.tocoo()
    diff = Y[A.row] - Y[A.col]  # (nnz, h, p)
    return np.einsum("e,ehp,ejp->hj", A.data, diff, diff)


def _mode_scatter(U_membership: np.ndarray, m_f: float, projected: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:

    Um = U_membership ** m_f
    s = Um.sum(axis=1)  # (n,)
    w = Um.sum(axis=0)  # (c,)
    safe_w = np.where(w > 0, w, 1.0)

    Ybar = projected.mean(axis=0)               # == Mbar @ V  (linearity of projection)
    Yc = projected - Ybar[None]
    Nk = np.einsum("nc,nhp->chp", Um, Yc) / safe_w[:, None, None]  # (Mk^f - Mbar) @ V

    Sw = (np.einsum("n,nhp,njp->hj", s, Yc, Yc)
          - np.einsum("c,chp,cjp->hj", w, Nk, Nk))
    Sb = np.einsum("c,chp,cjp->hj", w, Nk, Nk)
    return 0.5 * (Sw + Sw.T), 0.5 * (Sb + Sb.T)


def row_mode_matrices(images: np.ndarray, U_membership: np.ndarray, m_f: float,
                       V: np.ndarray, A_list, ridge: float = 0.0):

    Y = np.einsum("nhw,wp->nhp", images, V)  # Y_i = X_i V
    Sw, Sb = _mode_scatter(U_membership, m_f, Y)
    if ridge > 0:
        Sw = Sw + ridge * np.eye(Sw.shape[0])
    B_list = [graph_weighted_scatter(A, Y) for A in A_list]
    return Sw, Sb, B_list


def col_mode_matrices(images: np.ndarray, U_membership: np.ndarray, m_f: float,
                       U: np.ndarray, A_list, ridge: float = 0.0):

    images_T = np.transpose(images, (0, 2, 1))
    return row_mode_matrices(images_T, U_membership, m_f, U, A_list, ridge=ridge)


class FRNDA2D(BaseEstimator, TransformerMixin):


    def __init__(self, p1_components: int = 4, p2_components: int = 4, K_f: int = 5,
                 m_f: float = 2.0, scales=(3, 5, 7, 9, 11), lam: float = 1e-2,
                 gamma: float = 1e-2, ridge: float = 1e-6,
                 measure: Union[str, FuzzyMeasure] = "two-additive",
                 mu_pairwise_scale: float = 0.5, max_iter: int = 50, tol: float = 1e-6,
                 outer_max_iter: int = 10, outer_tol: float = 1e-6,
                 random_state: Optional[int] = None):
        self.p1_components = p1_components
        self.p2_components = p2_components
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
        self.outer_max_iter = outer_max_iter
        self.outer_tol = outer_tol
        self.random_state = random_state

    def _resolve_measure(self, M: int) -> FuzzyMeasure:
        if isinstance(self.measure, FuzzyMeasure):
            return self.measure
        rng = np.random.default_rng(self.random_state)
        if self.measure == "additive":
            return AdditiveMeasure.uniform(M)
        if self.measure == "two-additive":
            return TwoAdditiveMeasure.random_supermodular(
                M, rng=rng, pairwise_scale=self.mu_pairwise_scale)
        raise ValueError(f"Unknown measure spec: {self.measure!r}")

    def _joint_objective(self, images, U_mem, U, V, A_list, measure) -> float:

        Sw, Sb, B_list = row_mode_matrices(images, U_mem, self.m_f, V, A_list,
                                            ridge=self.ridge)
        return trace_ratio_objective(U, Sw, Sb, B_list, measure,
                                      self.lam, self.gamma, self.ridge)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FRNDA2D":

        images = np.asarray(X, dtype=float)
        if images.ndim != 3:
            raise ValueError("FRNDA2D expects X of shape (n_samples, h, w); "
                              f"got ndim={images.ndim}.")
        y = np.asarray(y)
        n, h, w = images.shape
        p1 = min(self.p1_components, h)
        p2 = min(self.p2_components, w)

        flat = images.reshape(n, h * w)
        U_mem, classes = fuzzy_knn_membership(flat, y, self.K_f, self.m_f)   # Frobenius == Euclidean on flattening
        graphs = build_multi_scale_graphs(flat, self.scales)                  
        measure = self._resolve_measure(len(graphs.scales))


        stacked = images.reshape(n * h, w)
        _, _, Vt = np.linalg.svd(stacked, full_matrices=False)
        V = orthonormalize(Vt[:p2].T)

        history = []
        U = None
        converged = False
        s = 0
        J_prev = None
        for s in range(1, self.outer_max_iter + 1):
            Sw_r, Sb_r, B_r = row_mode_matrices(images, U_mem, self.m_f, V, graphs.A,
                                                 ridge=self.ridge)
            res_u = frnda_optimize(Sw_r, Sb_r, B_r, measure, p=p1,
                                    lam=self.lam, gamma=self.gamma, eps=self.ridge,
                                    max_iter=self.max_iter, tol=self.tol)
            U = res_u.W                                                       

            Sw_c, Sb_c, B_c = col_mode_matrices(images, U_mem, self.m_f, U, graphs.A,
                                                 ridge=self.ridge)
            res_v = frnda_optimize(Sw_c, Sb_c, B_c, measure, p=p2,
                                    lam=self.lam, gamma=self.gamma, eps=self.ridge,
                                    max_iter=self.max_iter, tol=self.tol)
            V = res_v.W                                                       

            J_curr = self._joint_objective(images, U_mem, U, V, graphs.A, measure)
            history.append(J_curr)
            if J_prev is not None and abs(J_curr - J_prev) < self.outer_tol:
                converged = True
                J_prev = J_curr
                break
            J_prev = J_curr

        self.U_ = U
        self.V_ = V
        self.U_membership_ = U_mem
        self.classes_ = classes
        self.graphs_ = graphs
        self.measure_ = measure
        self.history_ = history
        self.n_iter_ = s
        self.converged_ = converged
        self.image_shape_ = (h, w)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        check_is_fitted(self, "U_")
        images = np.asarray(X, dtype=float)
        squeeze = images.ndim == 2
        if squeeze:
            images = images[None]
        Y = np.einsum("hp,nhw,wq->npq", self.U_, images, self.V_)
        return Y[0] if squeeze else Y

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        return self.fit(X, y).transform(X)
