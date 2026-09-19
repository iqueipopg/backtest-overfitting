"""Monte Carlo check of the formulas: does the closed-form expected maximum
Sharpe of N zero-skill trials match simulation, and does the DSR keep the
false-positive rate near its nominal level while the naive PSR does not?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .psr import deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio


def simulate_max_sharpe(
    n_trials_grid: tuple[int, ...] = (2, 5, 10, 20, 50, 100, 200, 500),
    n_obs: int = 500,
    n_sims: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """For each N: empirical mean of the maximum per-period Sharpe over
    ``n_sims`` draws of N iid normal zero-mean trials, the closed-form
    expectation using the *observed* cross-trial variance of Sharpe estimates,
    and the false-positive rates of ``PSR >= 0.95`` and ``DSR >= 0.95`` for
    the selected trial."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_trials_grid:
        max_sr, e_theory, fp_psr, fp_dsr = [], [], 0, 0
        for _ in range(n_sims):
            x = rng.standard_normal((n_obs, n))
            sr = x.mean(0) / x.std(0, ddof=1)
            best = int(np.argmax(sr))
            var_sr = sr.var(ddof=1) if n > 1 else 1.0 / n_obs
            max_sr.append(sr[best])
            e_theory.append(expected_max_sharpe(n, var_sr))
            fp_psr += probabilistic_sharpe_ratio(sr[best], 0.0, n_obs) >= 0.95
            dsr, _ = deflated_sharpe_ratio(sr[best], var_sr, n, n_obs)
            fp_dsr += dsr >= 0.95
        rows.append(
            {
                "n_trials": n,
                "n_obs": n_obs,
                "n_sims": n_sims,
                "max_sharpe_simulated": float(np.mean(max_sr)),
                "max_sharpe_simulated_se": float(np.std(max_sr, ddof=1) / np.sqrt(n_sims)),
                "max_sharpe_theory": float(np.mean(e_theory)),
                "false_positive_psr": fp_psr / n_sims,
                "false_positive_dsr": fp_dsr / n_sims,
            }
        )
    return pd.DataFrame(rows)
