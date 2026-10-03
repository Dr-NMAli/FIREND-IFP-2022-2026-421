

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .choquet import FuzzyMeasure, choquet_integral
from .objective import scale_regularizers, trace_ratio_objective
from .utils import generalized_top_eigh, irls_weights, l21_norm_smoothed, top_eigh


@dataclass
class OptimizeResult:


    W: np.ndarray
    history: List[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def frnda_optimize(Sw: np.ndarray, Sb: np.ndarray, B_list: Sequence[np.ndarray],
                    measure: FuzzyMeasure, p: int, lam: float = 1e-2, gamma: float = 1e-2,
                    eps: float = 1e-6, max_iter: int = 50, tol: float = 1e-6,
                    W_init: Optional[np.ndarray] = None) -> OptimizeResult:



    d = Sw.shape[0]
    p = int(min(p, d))

    if W_init is None:
        _, W = generalized_top_eigh(Sb, Sw, p)  # Algorithm 1, line 4
    else:
        Q, _ = np.linalg.qr(np.asarray(W_init, dtype=float))
        W = Q[:, :p]

    history = [trace_ratio_objective(W, Sw, Sb, B_list, measure, lam, gamma, eps)]
    J_prev = history[0]
    converged = False
    t = 0

    for t in range(1, max_iter + 1):
        r = scale_regularizers(W, B_list)                                    
        R_choquet, c = choquet_integral(r, measure, return_coeffs=True)      
        D_W = irls_weights(W, eps)                                           

        C = Sw + lam * np.diag(D_W) + gamma * sum(cm * B for cm, B in zip(c, B_list))  
        C = 0.5 * (C + C.T)

        num = float(np.trace(W.T @ Sb @ W))
        den = (float(np.trace(W.T @ Sw @ W))
               + lam * l21_norm_smoothed(W, eps)
               + gamma * R_choquet)
        rho = num / den                                                      

        _, W_new = top_eigh(Sb - rho * C, p)                                 
        W = W_new

        J_new = trace_ratio_objective(W, Sw, Sb, B_list, measure, lam, gamma, eps)
        history.append(J_new)

        if abs(J_new - J_prev) < tol:
            converged = True
            J_prev = J_new
            break
        J_prev = J_new

    return OptimizeResult(W=W, history=history, n_iter=t, converged=converged)
