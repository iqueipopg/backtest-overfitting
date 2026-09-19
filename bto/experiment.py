"""End-to-end experiment: build a strategy zoo on one asset, pick the best
backtest, and ask whether it survives selection-bias corrections.

Outputs ``results/summary.json``, ``results/trials.csv`` and four figures in
``figures/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import __version__  # noqa: E402
from .cscv import cscv  # noqa: E402
from .data import load_prices  # noqa: E402
from .metrics import annualize_sharpe, higher_moments, sharpe_ratio  # noqa: E402
from .psr import (  # noqa: E402
    deflated_sharpe_ratio,
    effective_number_of_trials,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from .strategies import build_trials  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PPY = 252


def run(ticker: str = "SPY", start: str = "1995-01-01", cost_bps: float = 0.0,
        n_splits: int = 16, refresh: bool = False, out: Path | None = None) -> dict:
    out = out or ROOT
    fig_dir = out / "figures"
    res_dir = out / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(ticker, start, refresh)
    trials_df, trials = build_trials(prices, cost_bps=cost_bps)
    X = trials_df.to_numpy()
    T, N = X.shape

    sr = np.array([sharpe_ratio(X[:, j]) for j in range(N)])
    best = int(np.argmax(sr))
    sr_best = float(sr[best])
    skew, kurt = higher_moments(X[:, best])
    var_sr = float(np.var(sr, ddof=1))
    n_eff = effective_number_of_trials(X)

    psr0 = probabilistic_sharpe_ratio(sr_best, 0.0, T, skew, kurt)
    dsr_raw, sr_star_raw = deflated_sharpe_ratio(sr_best, var_sr, N, T, skew, kurt)
    dsr_eff, sr_star_eff = deflated_sharpe_ratio(sr_best, var_sr, max(2, int(round(n_eff))), T, skew, kurt)
    mintrl_raw = min_track_record_length(sr_best, sr_star_raw, skew, kurt)
    mintrl_eff = min_track_record_length(sr_best, sr_star_eff, skew, kurt)
    minbtl_raw = min_backtest_length(N, annualize_sharpe(sr_best, PPY))
    minbtl_eff = min_backtest_length(max(2, int(round(n_eff))), annualize_sharpe(sr_best, PPY))

    cv = cscv(X, n_splits=n_splits)
    slope, intercept = cv.degradation

    bh = prices.pct_change().loc[trials_df.index]
    sr_bh = sharpe_ratio(bh.to_numpy())

    summary = {
        "version": __version__,
        "ticker": ticker,
        "sample_start": str(trials_df.index[0].date()),
        "sample_end": str(trials_df.index[-1].date()),
        "n_obs": int(T),
        "years": round(T / PPY, 2),
        "n_trials": int(N),
        "n_trials_effective": round(n_eff, 2),
        "cost_bps": cost_bps,
        "best_trial": trials[best].label,
        "best_sharpe_annual": round(annualize_sharpe(sr_best, PPY), 3),
        "buy_and_hold_sharpe_annual": round(annualize_sharpe(sr_bh, PPY), 3),
        "median_trial_sharpe_annual": round(annualize_sharpe(float(np.median(sr)), PPY), 3),
        "best_skew": round(skew, 3),
        "best_kurtosis": round(kurt, 3),
        "psr_vs_zero": round(psr0, 4),
        "expected_max_sharpe_annual_rawN": round(annualize_sharpe(sr_star_raw, PPY), 3),
        "expected_max_sharpe_annual_effN": round(annualize_sharpe(sr_star_eff, PPY), 3),
        "dsr_rawN": round(dsr_raw, 4),
        "dsr_effN": round(dsr_eff, 4),
        "min_track_record_years_rawN": None if np.isinf(mintrl_raw) else round(mintrl_raw / PPY, 1),
        "min_track_record_years_effN": None if np.isinf(mintrl_eff) else round(mintrl_eff / PPY, 1),
        "min_backtest_years_rawN": round(minbtl_raw, 1),
        "min_backtest_years_effN": round(minbtl_eff, 1),
        "cscv_splits": n_splits,
        "cscv_combinations": cv.n_combinations,
        "pbo": round(cv.pbo, 4),
        "prob_oos_loss": round(cv.prob_oos_loss, 4),
        "degradation_slope": round(slope, 3),
        "degradation_intercept_annual": round(annualize_sharpe(intercept, PPY), 3),
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    table = pd.DataFrame({
        "trial": [t.label for t in trials],
        "family": [t.family for t in trials],
        "sharpe_annual": annualize_sharpe(sr, PPY),
    }).sort_values("sharpe_annual", ascending=False)
    table.to_csv(res_dir / "trials.csv", index=False)

    _plot_heatmap(trials, sr, fig_dir / "sharpe_heatmap.png", ticker)
    _plot_trial_distribution(sr, sr_star_raw, sr_star_eff, best, fig_dir / "trial_sharpes.png", trials)
    _plot_cscv(cv, fig_dir / "cscv_logits.png", fig_dir / "cscv_degradation.png")
    _plot_equity(trials_df.iloc[:, best], bh, trials[best].label, fig_dir / "equity_best_vs_bh.png")
    return summary


def _plot_heatmap(trials, sr, path, ticker):
    ma = [(t.params[0], t.params[1], s) for t, s in zip(trials, sr) if t.family == "ma_crossover"]
    fasts = sorted({f for f, _, _ in ma})
    slows = sorted({s for _, s, _ in ma})
    grid = np.full((len(fasts), len(slows)), np.nan)
    for f, s, v in ma:
        grid[fasts.index(f), slows.index(s)] = annualize_sharpe(v, PPY)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    vmax = np.nanmax(np.abs(grid))
    im = ax.imshow(grid, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto", origin="lower")
    ax.set_xticks(range(len(slows)), slows)
    ax.set_yticks(range(len(fasts)), fasts)
    ax.set_xlabel("slow SMA window (days)")
    ax.set_ylabel("fast SMA window (days)")
    ax.set_title(f"In-sample annualised Sharpe, long/short SMA crossover on {ticker}")
    for i in range(len(fasts)):
        for j in range(len(slows)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Sharpe (annual)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_trial_distribution(sr, sr_star_raw, sr_star_eff, best, path, trials):
    a = annualize_sharpe(sr, PPY)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(a, bins=30, color="#9ecae1", edgecolor="white")
    ax.axvline(a[best], color="#d62728", lw=2, label=f"best trial {trials[best].label}: {a[best]:.2f}")
    ax.axvline(annualize_sharpe(sr_star_raw, PPY), color="black", ls="--",
               label=f"E[max SR] under null, N={len(sr)}: {annualize_sharpe(sr_star_raw, PPY):.2f}")
    ax.axvline(annualize_sharpe(sr_star_eff, PPY), color="grey", ls=":",
               label=f"E[max SR] under null, N_eff: {annualize_sharpe(sr_star_eff, PPY):.2f}")
    ax.set_xlabel("annualised Sharpe ratio of each trial")
    ax.set_ylabel("number of trials")
    ax.set_title("Where does the best backtest sit relative to pure luck?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cscv(cv, path_logits, path_degradation):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cv.logits, bins=40, color="#9ecae1", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("logit of the OOS relative rank of the IS-best trial")
    ax.set_ylabel("number of IS/OOS partitions")
    ax.set_title(f"CSCV, S={cv.n_splits} ({cv.n_combinations:,} partitions): PBO = {cv.pbo:.2f}")
    fig.tight_layout()
    fig.savefig(path_logits, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = annualize_sharpe(cv.sr_is_best, PPY)
    y = annualize_sharpe(cv.sr_oos_best, PPY)
    ax.scatter(x, y, s=4, alpha=0.25, color="#3182bd")
    slope, intercept = cv.degradation
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, slope * xs + annualize_sharpe(intercept, PPY), color="#d62728",
            label=f"OLS slope = {slope:.2f}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("in-sample Sharpe of the IS-best trial (annual)")
    ax.set_ylabel("out-of-sample Sharpe of that trial (annual)")
    ax.set_title(f"Performance degradation; P[OOS loss] = {cv.prob_oos_loss:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_degradation, dpi=150)
    plt.close(fig)


def _plot_equity(best_returns, bh_returns, label, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot((1 + best_returns).cumprod(), label=f"best trial in-sample: {label}", color="#d62728")
    ax.plot((1 + bh_returns).cumprod(), label="buy and hold", color="#636363")
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 unit (log scale)")
    ax.set_title("Best in-sample trial (long/short) against buy and hold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Deflated Sharpe and PBO on a strategy zoo")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="1995-01-01")
    p.add_argument("--cost-bps", type=float, default=0.0, help="one-way cost per unit turnover")
    p.add_argument("--splits", type=int, default=16, help="CSCV blocks (even)")
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    p.add_argument("--out", default=None, help="output root (default: repository root)")
    a = p.parse_args(argv)
    summary = run(a.ticker, a.start, a.cost_bps, a.splits, a.refresh,
                  Path(a.out) if a.out else None)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
