
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def make_overlap_synth(n_samples: int = 750, n_classes: int = 5, n_features: int = 100,
                        n_informative: int = 10, overlap: float = 1.0,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:


    rng = np.random.default_rng(random_state)
    n_informative = min(n_informative, n_features)

    directions = rng.normal(size=(n_classes, n_informative))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions /= norms
    radius = 5.0 / max(overlap, 1e-3)
    means = directions * radius

    counts = np.full(n_classes, n_samples // n_classes)
    counts[: n_samples - counts.sum()] += 1
    y = np.repeat(np.arange(n_classes), counts)
    rng.shuffle(y)

    X = np.zeros((n_samples, n_features))
    X[:, :n_informative] = means[y] + rng.normal(size=(n_samples, n_informative))
    if n_features > n_informative:
        X[:, n_informative:] = rng.normal(size=(n_samples, n_features - n_informative))
    return X, y


def make_isotropic_gaussian_cloud(n_samples: int = 1000, n_features: int = 3,
                                   random_state: Optional[int] = None) -> np.ndarray:

    rng = np.random.default_rng(random_state)
    return rng.normal(size=(n_samples, n_features))


def add_gaussian_pixel_noise(X: np.ndarray, sigma: float, value_range=(0.0, 1.0),
                              random_state: Optional[int] = None) -> np.ndarray:

    rng = np.random.default_rng(random_state)
    noisy = X + rng.normal(scale=sigma, size=X.shape)
    lo, hi = value_range
    return np.clip(noisy, lo, hi)
