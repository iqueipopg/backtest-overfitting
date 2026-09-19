"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012, 2014).

Everything here works in per-period Sharpe units. ``n_obs`` is the number of
return observations in the backtest; ``skew`` and ``kurt`` are the third and
fourth standardised moments of the *selected* strategy's returns (``kurt`` = 3
for a normal distribution).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def _sr_std_factor(sr: float, skew: float, kurt: float) -> float:
    """``sqrt(1 - g3*SR + (g4-1)/4 * SR^2)``: the non-normality adjustment of the
    standard error of the Sharpe estimator (Lo, 2002; Mertens, 2002)."""
    val = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if val <= 0:
        raise ValueError("Non-positive variance factor: check skew/kurtosis inputs.")
    return float(np.sqrt(val))


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n_obs: int, skew: float = 0.0, kurt: float = 3.0
) -> float:
    """``PSR(SR*) = Prob[true SR > SR*]`` given the observed estimate.

    ``PSR = Phi[ (SR - SR*) * sqrt(T - 1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2) ]``
    """
    if n_obs < 2:
        raise ValueError("n_obs must be >= 2")
    z = (sr - sr_benchmark) * np.sqrt(n_obs - 1) / _sr_std_factor(sr, skew, kurt)
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr_trials: float) -> float:
    """Expected maximum Sharpe among ``n_trials`` independent zero-skill trials
    whose Sharpe estimates have variance ``var_sr_trials``.

    ``E[max SR] ~ sqrt(V) * [ (1 - g) Z^{-1}(1 - 1/N) + g Z^{-1}(1 - 1/(N e)) ]``
    with ``g`` the Euler-Mascheroni constant. For ``N = 1`` there is no selection.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_sr_trials < 0:
        raise ValueError("var_sr_trials must be >= 0")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr_trials) * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    sr_best: float,
    var_sr_trials: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio of the best trial.

    Returns ``(dsr, sr_star)``: the PSR of the selected strategy against the
    benchmark ``sr_star = E[max SR]`` implied by the number and dispersion of
    trials. A DSR below 0.95 means the best backtest is not distinguishable, at
    the 5% level, from the luckiest of ``N`` noise strategies.
    """
    sr_star = expected_max_sharpe(n_trials, var_sr_trials)
    return probabilistic_sharpe_ratio(sr_best, sr_star, n_obs, skew, kurt), sr_star


def min_track_record_length(
    sr: float,
    sr_benchmark: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    alpha: float = 0.05,
) -> float:
    """Number of observations needed for ``PSR(SR*) >= 1 - alpha``.

    ``MinTRL = 1 + [1 - g3 SR + (g4 - 1)/4 SR^2] * ( Z_{1-alpha} / (SR - SR*) )^2``
    Returns ``inf`` when ``SR <= SR*``.
    """
    if sr <= sr_benchmark:
        return float("inf")
    z = norm.ppf(1.0 - alpha)
    return float(1.0 + _sr_std_factor(sr, skew, kurt) ** 2 * (z / (sr - sr_benchmark)) ** 2)


def min_backtest_length(n_trials: int, target_sharpe_annual: float) -> float:
    """Minimum backtest length, in *years*, such that the expected maximum
    annualised Sharpe of ``N`` zero-skill trials stays below
    ``target_sharpe_annual`` (Bailey, Borwein, Lopez de Prado & Zhu, 2014).

    Under iid returns the annualised Sharpe estimate over ``y`` years has
    standard deviation about ``1/sqrt(y)``, hence

        ``MinBTL = ( [(1-g) Z^{-1}(1-1/N) + g Z^{-1}(1-1/(Ne))] / SR_target )^2``

    which is bounded above by ``2 ln N / SR_target^2``.
    """
    if target_sharpe_annual <= 0:
        raise ValueError("target_sharpe_annual must be > 0")
    return float((expected_max_sharpe(n_trials, 1.0) / target_sharpe_annual) ** 2)


def effective_number_of_trials(trial_returns: np.ndarray) -> float:
    """Participation-ratio estimate of the number of *independent* trials.

    Trials generated from one strategy family are highly correlated, so the raw
    count ``N`` overstates the selection pool. From the eigenvalues ``l_i`` of the
    trial correlation matrix, ``N_eff = (sum l_i)^2 / sum l_i^2``: equal to ``N``
    for uncorrelated trials and to 1 for perfectly correlated ones. Bailey &
    Lopez de Prado (2014) recommend clustering; this is a cheap, deterministic
    proxy that gives the *lenient* end of the deflation.
    """
    x = np.asarray(trial_returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("trial_returns must be a 2-D array (T x N)")
    x = x[~np.isnan(x).any(axis=1)]
    if x.shape[1] < 2:
        return float(x.shape[1])
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    eig = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    return float(eig.sum() ** 2 / (eig**2).sum())
