"""Probability of Backtest Overfitting via Combinatorially Symmetric
Cross-Validation (Bailey, Borwein, Lopez de Prado & Zhu, 2017).

Given a ``T x N`` matrix of trial returns, the sample is split into ``S`` equal
blocks. For every one of the ``C(S, S/2)`` ways of choosing half the blocks as
in-sample (IS), the IS-best trial is selected and its *relative rank* in the
complementary out-of-sample (OOS) half is recorded. Ranks are mapped to logits
``lambda = log(w / (1 - w))``; the PBO is the fraction of combinations where the
IS winner ends up in the bottom half OOS (``lambda <= 0``).

The implementation only needs per-block sums and sums of squares, so the whole
combinatorial sweep is two matrix products regardless of ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass
class CSCVResult:
    n_splits: int
    n_combinations: int
    logits: np.ndarray  # lambda_c for each combination
    sr_is_best: np.ndarray  # IS Sharpe of the IS-best trial, per combination
    sr_oos_best: np.ndarray  # OOS Sharpe of that same trial, per combination
    best_index: np.ndarray  # index of the IS-best trial, per combination

    @property
    def pbo(self) -> float:
        """Probability of backtest overfitting: ``P[lambda <= 0]``."""
        return float(np.mean(self.logits <= 0.0))

    @property
    def prob_oos_loss(self) -> float:
        """Fraction of combinations where the IS-best trial loses money OOS."""
        return float(np.mean(self.sr_oos_best < 0.0))

    @property
    def degradation(self) -> tuple[float, float]:
        """OLS ``(slope, intercept)`` of OOS Sharpe on IS Sharpe of the winner.

        A slope near or below zero means IS performance does not carry over."""
        slope, intercept = np.polyfit(self.sr_is_best, self.sr_oos_best, 1)
        return float(slope), float(intercept)


def _block_stats(returns: np.ndarray, n_splits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks = np.array_split(np.arange(returns.shape[0]), n_splits)
    sums = np.vstack([returns[b].sum(axis=0) for b in blocks])
    sq = np.vstack([(returns[b] ** 2).sum(axis=0) for b in blocks])
    counts = np.array([len(b) for b in blocks], dtype=float)
    return sums, sq, counts


def _sharpe_from_sums(s: np.ndarray, q: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Vectorised per-period Sharpe from sums ``s``, sums of squares ``q`` and
    observation counts ``n`` (``n`` broadcast along columns)."""
    n = n[:, None]
    mean = s / n
    var = (q / n - mean**2) * n / (n - 1.0)
    var = np.clip(var, 1e-300, None)
    return mean / np.sqrt(var)


def cscv(returns: np.ndarray, n_splits: int = 16) -> CSCVResult:
    """Run CSCV on a ``T x N`` matrix of per-period trial returns.

    ``n_splits`` must be even; the default 16 gives 12,870 IS/OOS partitions.
    Rows containing NaN (warm-up periods) are dropped before splitting.
    """
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be an even integer >= 2")
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a 2-D array (T x N)")
    x = x[~np.isnan(x).any(axis=1)]
    t, n = x.shape
    if n < 2:
        raise ValueError("need at least two trials")
    if t < 2 * n_splits:
        raise ValueError("too few observations for the requested number of splits")

    sums, sq, counts = _block_stats(x, n_splits)
    combos = np.array(list(combinations(range(n_splits), n_splits // 2)))
    mask_is = np.zeros((combos.shape[0], n_splits))
    mask_is[np.arange(combos.shape[0])[:, None], combos] = 1.0
    mask_oos = 1.0 - mask_is

    sr_is = _sharpe_from_sums(mask_is @ sums, mask_is @ sq, mask_is @ counts)
    sr_oos = _sharpe_from_sums(mask_oos @ sums, mask_oos @ sq, mask_oos @ counts)

    best = np.argmax(sr_is, axis=1)
    rows = np.arange(combos.shape[0])
    # relative OOS rank of the IS winner: 1/(N+1) worst ... N/(N+1) best
    rank = (sr_oos < sr_oos[rows, best][:, None]).sum(axis=1) + 1
    omega = rank / (n + 1.0)
    logits = np.log(omega / (1.0 - omega))

    return CSCVResult(
        n_splits=n_splits,
        n_combinations=int(combos.shape[0]),
        logits=logits,
        sr_is_best=sr_is[rows, best],
        sr_oos_best=sr_oos[rows, best],
        best_index=best,
    )
