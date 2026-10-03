
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

from .graph import k_occurrence_counts


def hubness_report(X: np.ndarray, k: int = 10) -> Dict[str, object]:


    counts = k_occurrence_counts(X, k)
    return {
        "k": k,
        "k_occurrence": counts,
        "mean": float(counts.mean()),
        "skewness": float(stats.skew(counts)),
        "n_hubs_gt_3k": int(np.sum(counts > 3 * k)),
        "n_orphans": int(np.sum(counts == 0)),
    }
