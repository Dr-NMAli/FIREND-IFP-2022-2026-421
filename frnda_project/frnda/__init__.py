"""
FRNDA -- Fuzzy Reverse-Neighborhood Discriminant Analysis
===========================================================

A from-scratch Python implementation of:

Quick start
-----------
>>> from frnda import FRNDA
>>> model = FRNDA(n_components=4, K_f=5, m_f=2.0, random_state=0)
>>> Z = model.fit_transform(X, y)     # X: (n, d), y: (n,)
>>> model.history_[-1] >= model.history_[0]   # monotone objective "convergence")
True


"""
from .choquet import AdditiveMeasure, FuzzyMeasure, TwoAdditiveMeasure, choquet_integral
from .frnda import FRNDA
from .frnda2d import FRNDA2D
from .rnda import RNDA

__all__ = [
    "FRNDA",
    "FRNDA2D",
    "RNDA",
    "FuzzyMeasure",
    "AdditiveMeasure",
    "TwoAdditiveMeasure",
    "choquet_integral",
]

__version__ = "0.1.0"
