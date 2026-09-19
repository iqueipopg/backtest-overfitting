import numpy as np
import pandas as pd
import pytest

from bto import build_trials, ma_crossover, ts_momentum


def _prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))), index=idx)


def test_positions_are_lagged_no_lookahead():
    p = _prices()
    r = p.pct_change()
    strat = ma_crossover(p, 5, 20)
    sma_f, sma_s = p.rolling(5).mean(), p.rolling(20).mean()
    pos = np.sign(sma_f - sma_s)
    # today's strategy return uses yesterday's signal
    t = 100
    assert strat.iloc[t] == pytest.approx(pos.iloc[t - 1] * r.iloc[t])


def test_long_short_positions_only():
    p = _prices()
    r = p.pct_change()
    strat = ma_crossover(p, 10, 50).dropna()
    ratio = (strat / r.loc[strat.index]).round(6)
    assert set(ratio.dropna().unique()) <= {1.0, -1.0}


def test_costs_reduce_returns_on_turnover():
    p = _prices()
    free = ts_momentum(p, 20).dropna()
    paid = ts_momentum(p, 20, cost_bps=10).dropna()
    assert (paid <= free + 1e-12).all()
    assert paid.sum() < free.sum()


def test_invalid_windows():
    with pytest.raises(ValueError):
        ma_crossover(_prices(), 50, 20)


def test_build_trials_aligned_and_labelled():
    df, trials = build_trials(_prices(), fast=(5, 10), slow=(20, 50), lookback=(10, 60))
    assert df.shape[1] == len(trials) == 6
    assert not df.isna().any().any()
    assert [t.label for t in trials] == ["MA(5,20)", "MA(5,50)", "MA(10,20)", "MA(10,50)", "TSMOM(10)", "TSMOM(60)"]
