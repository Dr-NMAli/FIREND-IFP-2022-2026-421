"""
Generic numerical linear-algebra helpers shared across the FRNDA package.

These are small, well-tested pieces of linear algebra (orthonormalization, symmetric
and generalized eigenproblems, the row-sparsity norm and its IRLS
majorizer) that are reused throughout ``fuzzy_membership.py``,
``scatter.py``, ``objective.py`` and ``optimize.py``.

Data convention
----------------
Stores samples as columns of X in R^{d x n}. Throughout this
package we instead use the scikit-learn convention X in R^{n x d} (rows are
samples), which is more idiomatic Python/NumPy. .
"""
from __future__ import annotations

import numpy as np
from scipy import linalg as sla


def pairwise_sq_dists(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Squared Euclidean distances between rows of X and rows of Y.

    Parameters
    ----------
    X : (n, d) array
    Y : (m, d) array, optional. Defaults to X.

    Returns
    -------
    D2 : (n, m) array with D2[i, j] = ||X[i] - Y[j]||^2.
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    sq_x = np.einsum("ij,ij->i", X, X)
    sq_y = np.einsum("ij,ij->i", Y, Y)
    D2 = sq_x[:, None] + sq_y[None, :] - 2.0 * (X @ Y.T)
    np.maximum(D2, 0.0, out=D2)  # clip tiny negative round-off
    return D2


def row_l2_norms(W: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean norms ||w^i||_2, i = 1..d (the building block of
    the l2,1 norm, )."""
    return np.sqrt(np.einsum("ij,ij->i", W, W))


def l21_norm(W: np.ndarray) -> float:
    r"""The (non-smooth) row-sparsity norm ``||W||_{2,1} = sum_i ||w^i||_2``
    (obj_fuzzy)."""
    return float(np.sum(row_l2_norms(W)))


def l21_norm_smoothed(W: np.ndarray, eps: float) -> float:
    r"""The smoothed row-sparsity norm
    ``||W||_{2,1,eps} = sum_i sqrt(||w^i||_2^2 + eps)``.

    This is the surrogate whose IRLS majorization is exploited.
    """
    row_sq = np.einsum("ij,ij->i", W, W)
    return float(np.sum(np.sqrt(row_sq + eps)))


def irls_weights(W: np.ndarray, eps: float) -> np.ndarray:
    r"""Diagonal IRLS majorizer weights for the l2,1 penalty:

        D_W = diag( 1 / (2 sqrt(||w^i||_2^2 + eps)) )_{i=1}^d

    ("l2,1 majorization").
    """
    row_sq = np.einsum("ij,ij->i", W, W)
    return 1.0 / (2.0 * np.sqrt(row_sq + eps))


def orthonormalize(W: np.ndarray) -> np.ndarray:
    """An orthonormal basis (thin QR) spanning the column space of W, with a
    deterministic sign convention (non-negative diagonal of R)."""
    W = np.asarray(W, dtype=float)
    Q, R = np.linalg.qr(W)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs


def top_eigh(M: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-p eigenpairs of a symmetric matrix M, sorted by decreasing
    eigenvalue. Returns (eigenvalues, eigenvectors) with eigenvectors as
    columns; since M is symmetric, the eigenvectors are automatically
    orthonormal (W^T W = I_p) -- the property Algorithm 1 relies on at
    every outer iteration ("top-p eigenvectors of Sb - rho C").
    """
    M = 0.5 * (M + M.T)  # enforce exact symmetry against round-off
    vals, vecs = np.linalg.eigh(M)
    idx = np.argsort(vals)[::-1][:p]
    return vals[idx], vecs[:, idx]


def generalized_top_eigh(A: np.ndarray, B: np.ndarray, p: int,
                          ridge: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Top-p generalized eigenpairs of the pencil (A, B): A v = lambda B v,
    with B symmetric positive definite.

    Used once, to warm-start W^(0) from the generalized eigenvectors of
    (S_b^f, S_w^f) -- i.e. an ordinary (fuzzy) LDA solution -- per
    ``scipy.linalg.eigh(A, B)`` returns B-orthonormal
    eigenvectors (v_i^T B v_j = delta_ij), not I-orthonormal ones, so we
    re-orthonormalize in the ordinary sense afterwards, "orthonormalize" step.
    """
    if ridge > 0:
        B = B + ridge * np.eye(B.shape[0])
    vals, vecs = sla.eigh(A, B)  # ascending eigenvalues
    idx = np.argsort(vals)[::-1][:p]
    W = vecs[:, idx]
    return vals[idx], orthonormalize(W)
