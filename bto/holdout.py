"""Plain temporal holdout: select the best trial on the first part of the
sample, evaluate that same trial on the rest. The simplest check a selection
procedure must pass, and the one CSCV generalises."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import higher_moments, sharpe_ratio
from .psr import probabilistic_sharpe_ratio


@dataclass
class HoldoutResult:
    split_index: int
    best_index: int
    sr_is_best: float  # per period
    sr_oos_best: float
    oos_rank: float  # relative rank of the IS winner among trials OOS, in (0, 1)
    psr_oos_vs_zero: float
    sr_oos_median_trial: float


def holdout(trial_returns: np.ndarray, split: float = 0.7) -> HoldoutResult:
    """``split`` is the in-sample fraction of the observations (time ordered)."""
    x = np.asarray(trial_returns, dtype=float)
    x = x[~np.isnan(x).any(axis=1)]
    t, n = x.shape
    if not 0.1 <= split <= 0.9 or t < 20:
        raise ValueError("split must be in [0.1, 0.9] and the sample long enough")
    k = int(t * split)
    sr_is = np.array([sharpe_ratio(x[:k, j]) for j in range(n)])
    sr_oos = np.array([sharpe_ratio(x[k:, j]) for j in range(n)])
    best = int(np.nanargmax(sr_is))
    skew, kurt = higher_moments(x[k:, best])
    rank = ((sr_oos < sr_oos[best]).sum() + 1) / (n + 1.0)
    return HoldoutResult(
        split_index=k,
        best_index=best,
        sr_is_best=float(sr_is[best]),
        sr_oos_best=float(sr_oos[best]),
        oos_rank=float(rank),
        psr_oos_vs_zero=probabilistic_sharpe_ratio(sr_oos[best], 0.0, t - k, skew, kurt),
        sr_oos_median_trial=float(np.nanmedian(sr_oos)),
    )
