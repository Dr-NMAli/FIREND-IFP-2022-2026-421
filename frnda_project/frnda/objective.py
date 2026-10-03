

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

from .choquet import FuzzyMeasure, choquet_integral
from .utils import l21_norm_smoothed


def scale_regularizers(W: np.ndarray, B_list: Sequence[np.ndarray]) -> np.ndarray:

    return np.array([float(np.trace(W.T @ B @ W)) for B in B_list])


def choquet_regularizer(W: np.ndarray, B_list: Sequence[np.ndarray], measure: FuzzyMeasure,
                         return_coeffs: bool = False
                         ) -> Union[float, Tuple[float, np.ndarray]]:


    r = scale_regularizers(W, B_list)
    return choquet_integral(r, measure, return_coeffs=return_coeffs)


def trace_ratio_objective(W: np.ndarray, Sw: np.ndarray, Sb: np.ndarray,
                           B_list: Sequence[np.ndarray], measure: FuzzyMeasure,
                           lam: float, gamma: float, eps: float) -> float:


    numer = float(np.trace(W.T @ Sb @ W))
    denom = (float(np.trace(W.T @ Sw @ W))
             + lam * l21_norm_smoothed(W, eps)
             + gamma * choquet_regularizer(W, B_list, measure))
    return numer / denom
