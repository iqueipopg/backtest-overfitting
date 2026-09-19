"""A deliberately ordinary strategy zoo used to generate correlated trials.

Both families trade a single asset long/short (+1 / -1) so that the equity
premium does not flatter every configuration: what is being tested is *timing*
skill, which is what a parameter search usually claims to have found.
Positions are lagged one period before being applied to returns, so there is no
look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trial:
    family: str
    params: tuple
    label: str


def _apply_position(position: pd.Series, asset_returns: pd.Series, cost_bps: float) -> pd.Series:
    pos = position.shift(1)
    turnover = pos.diff().abs().fillna(0.0)
    return pos * asset_returns - turnover * cost_bps / 1e4


def ma_crossover(
    prices: pd.Series, fast: int, slow: int, cost_bps: float = 0.0
) -> pd.Series:
    """Long when the fast SMA is above the slow SMA, short otherwise."""
    if fast >= slow:
        raise ValueError("fast window must be shorter than slow window")
    rets = prices.pct_change()
    sma_f = prices.rolling(fast).mean()
    sma_s = prices.rolling(slow).mean()
    position = np.sign(sma_f - sma_s).replace(0.0, np.nan).ffill()
    position[sma_s.isna()] = np.nan  # keep warm-up as NaN, not flat
    return _apply_position(position, rets, cost_bps)


def ts_momentum(prices: pd.Series, lookback: int, cost_bps: float = 0.0) -> pd.Series:
    """Time-series momentum: long if the trailing ``lookback``-day return is
    positive, short otherwise."""
    rets = prices.pct_change()
    trailing = prices.pct_change(lookback)
    position = np.sign(trailing).replace(0.0, np.nan).ffill()
    position[trailing.isna()] = np.nan
    return _apply_position(position, rets, cost_bps)


DEFAULT_FAST = (2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50)
DEFAULT_SLOW = (20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250)
DEFAULT_LOOKBACK = (5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 150, 180, 200, 250)


def build_trials(
    prices: pd.Series,
    fast: tuple[int, ...] = DEFAULT_FAST,
    slow: tuple[int, ...] = DEFAULT_SLOW,
    lookback: tuple[int, ...] = DEFAULT_LOOKBACK,
    cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, list[Trial]]:
    """Return a ``T x N`` DataFrame of trial returns and the matching metadata.

    The warm-up is aligned across trials: every column starts after the longest
    window so that all trials are evaluated on the same observations.
    """
    cols: dict[str, pd.Series] = {}
    trials: list[Trial] = []
    for f, s in product(fast, slow):
        if f >= s:
            continue
        label = f"MA({f},{s})"
        cols[label] = ma_crossover(prices, f, s, cost_bps)
        trials.append(Trial("ma_crossover", (f, s), label))
    for k in lookback:
        label = f"TSMOM({k})"
        cols[label] = ts_momentum(prices, k, cost_bps)
        trials.append(Trial("ts_momentum", (k,), label))
    df = pd.DataFrame(cols)
    df = df.dropna(how="any")
    return df, trials
