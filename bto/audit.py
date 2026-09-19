"""Audit an arbitrary ``T x N`` matrix of trial returns: how many trials,
how many independent ones, is the best one distinguishable from the
luckiest noise trial, and does the in-sample winner survive out of sample.

    python -m bto audit trials.csv --ppy 12
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clusters import cluster_trials
from .cscv import cscv
from .holdout import holdout
from .metrics import annualize_sharpe, higher_moments, sharpe_ratio
from .psr import deflated_sharpe_ratio, effective_number_of_trials, probabilistic_sharpe_ratio


def audit_trials(returns: pd.DataFrame, periods_per_year: int = 252, n_splits: int = 16, split: float = 0.7) -> dict:
    x = returns.dropna(how="any")
    labels = [str(c) for c in x.columns]
    arr = x.to_numpy(dtype=float)
    t, n = arr.shape
    sr = np.array([sharpe_ratio(arr[:, j]) for j in range(n)])
    best = int(np.nanargmax(sr))
    skew, kurt = higher_moments(arr[:, best])
    var_sr = float(np.nanvar(sr, ddof=1)) if n > 1 else 0.0
    n_pr = effective_number_of_trials(arr)
    n_cl, _ = cluster_trials(arr)
    ann = lambda v: float(annualize_sharpe(v, periods_per_year))  # noqa: E731
    out = {
        "n_obs": int(t),
        "n_trials": int(n),
        "n_effective_participation_ratio": round(float(n_pr), 2),
        "n_clusters": int(n_cl),
        "best_trial": labels[best],
        "best_sharpe_annual": round(ann(sr[best]), 3),
        "median_sharpe_annual": round(ann(float(np.nanmedian(sr))), 3),
        "psr_vs_zero": round(probabilistic_sharpe_ratio(sr[best], 0.0, t, skew, kurt), 4),
    }
    for tag, num in (("rawN", n), ("effN", max(1, int(round(n_pr)))), ("clusters", max(1, n_cl))):
        dsr, sr_star = deflated_sharpe_ratio(sr[best], var_sr, num, t, skew, kurt)
        out[f"expected_max_sharpe_annual_{tag}"] = round(ann(sr_star), 3)
        out[f"dsr_{tag}"] = round(float(dsr), 4)
    while n_splits > 2 and t < 2 * n_splits:
        n_splits -= 2
    if n >= 2:
        cv = cscv(arr, n_splits=n_splits)
        out.update(
            {
                "cscv_splits": int(cv.n_splits),
                "cscv_combinations": int(cv.n_combinations),
                "pbo": round(cv.pbo, 4),
                "prob_oos_loss": round(cv.prob_oos_loss, 4),
                "degradation_slope": round(cv.degradation[0], 3),
            }
        )
    if t >= 20:
        h = holdout(arr, split)
        out.update(
            {
                "holdout_split": split,
                "holdout_is_best_trial": labels[h.best_index],
                "holdout_is_sharpe_annual": round(ann(h.sr_is_best), 3),
                "holdout_oos_sharpe_annual": round(ann(h.sr_oos_best), 3),
                "holdout_oos_rank": round(h.oos_rank, 3),
                "holdout_oos_psr_vs_zero": round(h.psr_oos_vs_zero, 4),
                "holdout_oos_median_trial_sharpe_annual": round(ann(h.sr_oos_median_trial), 3),
            }
        )
    return out


def audit_csv(path: str | Path, periods_per_year: int = 252, n_splits: int = 16, split: float = 0.7) -> dict:
    df = pd.read_csv(path, index_col=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    return audit_trials(df, periods_per_year, n_splits, split)


def main(argv: list[str]) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="python -m bto audit", description=__doc__)
    p.add_argument("csv", help="T x N CSV of per-period trial returns; first column is the index")
    p.add_argument("--ppy", type=int, default=252, help="periods per year for annualisation (252 daily, 12 monthly)")
    p.add_argument("--splits", type=int, default=16)
    p.add_argument("--split", type=float, default=0.7, help="in-sample fraction for the holdout")
    a = p.parse_args(argv)
    print(json.dumps(audit_csv(a.csv, a.ppy, a.splits, a.split), indent=2))
