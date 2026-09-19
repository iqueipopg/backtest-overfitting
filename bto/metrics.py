"""Basic performance statistics.

All Sharpe ratios in this package are computed on *per-period* returns
(daily, for the experiments shipped here). The PSR/DSR formulas are defined in
per-period units with ``T`` observations; annualisation is only for display.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def sharpe_ratio(returns: np.ndarray, ddof: int = 1) -> float:
    """Per-period Sharpe ratio (mean / std), ignoring NaNs.

    Returns are assumed to be excess returns already. For the long/short timing
    strategies used in the experiments the funding leg nets out, so raw asset
    returns are a fair approximation.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if r.size < 2:
        return float("nan")
    sd = r.std(ddof=ddof)
    if sd == 0:
        return float("nan")
    return float(r.mean() / sd)


def annualize_sharpe(sr_per_period, periods_per_year: int = 252):
    """Scale a per-period Sharpe ratio (scalar or array) to annual units
    under the iid assumption."""
    out = np.asarray(sr_per_period, dtype=float) * np.sqrt(periods_per_year)
    return float(out) if out.ndim == 0 else out


def higher_moments(returns: np.ndarray) -> tuple[float, float]:
    """Return ``(skewness, kurtosis)`` of a return series.

    Kurtosis is the *non-excess* fourth standardised moment (3 for a normal),
    which is the convention used in Bailey & Lopez de Prado's PSR formula.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    return (
        float(stats.skew(r, bias=False)),
        float(stats.kurtosis(r, fisher=False, bias=False)),
    )
