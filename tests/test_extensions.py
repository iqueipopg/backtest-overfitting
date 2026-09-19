"""Clustering, holdout, Monte Carlo validation and the audit entry point."""

import json

import numpy as np
import pandas as pd
import pytest

from bto import audit_trials, cluster_trials, expected_max_sharpe, holdout, simulate_max_sharpe
from bto.audit import audit_csv
from bto.clusters import silhouette


def test_cluster_trials_recovers_block_structure():
    rng = np.random.default_rng(0)
    t = 3000
    blocks = [rng.standard_normal((t, 1)) for _ in range(3)]
    x = np.hstack([b + 0.3 * rng.standard_normal((t, 4)) for b in blocks])  # 3 groups of 4 clones
    k, labels = cluster_trials(x)
    assert k == 3
    for g in range(3):
        assert len(set(labels[4 * g : 4 * g + 4])) == 1


def test_cluster_trials_edge_cases():
    rng = np.random.default_rng(1)
    k, labels = cluster_trials(rng.standard_normal((100, 2)))
    assert k == 2 and list(labels) == [1, 2]
    d = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    assert silhouette(d, np.array([1, 1, 1])) == -1.0


def test_holdout_selects_skill_and_it_survives():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2000, 30)) * 0.01
    x[:, 7] += 0.002
    h = holdout(x, split=0.6)
    assert h.best_index == 7 and h.split_index == 1200
    assert h.sr_oos_best > 0 and h.oos_rank > 0.9 and h.psr_oos_vs_zero > 0.95
    with pytest.raises(ValueError):
        holdout(x, split=0.95)


def test_holdout_on_noise_winner_regresses_to_median():
    rng = np.random.default_rng(3)
    ranks = []
    for _ in range(40):
        x = rng.standard_normal((1000, 50))
        h = holdout(x)
        ranks.append(h.oos_rank)
    assert 0.35 < np.mean(ranks) < 0.65  # IS winner is average OOS


def test_simulation_matches_closed_form_and_dsr_controls_size():
    sim = simulate_max_sharpe(n_trials_grid=(5, 50), n_obs=400, n_sims=150, seed=4)
    for _, r in sim.iterrows():
        assert r["max_sharpe_simulated"] == pytest.approx(
            r["max_sharpe_theory"], abs=3 * r["max_sharpe_simulated_se"] + 0.01
        )
        assert r["false_positive_dsr"] <= 0.12
    assert sim.loc[sim.n_trials == 50, "false_positive_psr"].iloc[0] > 0.5
    assert expected_max_sharpe(50, 1 / 400) > expected_max_sharpe(5, 1 / 400)


def test_audit_trials_and_csv(tmp_path):
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.standard_normal((600, 12)) * 0.02, columns=[f"s{i}" for i in range(12)])
    rep = audit_trials(df, periods_per_year=12, n_splits=8)
    for key in ("n_trials", "n_clusters", "dsr_rawN", "dsr_clusters", "pbo", "holdout_oos_sharpe_annual"):
        assert key in rep
    assert rep["n_trials"] == 12 and rep["cscv_combinations"] == 70
    assert 0.0 <= rep["dsr_rawN"] <= 1.0
    path = tmp_path / "trials.csv"
    df.to_csv(path)
    rep2 = audit_csv(path, periods_per_year=12, n_splits=8)
    assert json.dumps(rep2) == json.dumps(rep)
