import numpy as np
import pytest

from bto import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)


def test_psr_is_one_half_at_benchmark():
    assert probabilistic_sharpe_ratio(0.05, 0.05, 1000) == pytest.approx(0.5)


def test_psr_increases_with_sample_size_and_sharpe():
    assert probabilistic_sharpe_ratio(0.05, 0.0, 2000) > probabilistic_sharpe_ratio(0.05, 0.0, 200)
    assert probabilistic_sharpe_ratio(0.10, 0.0, 500) > probabilistic_sharpe_ratio(0.05, 0.0, 500)


def test_psr_normal_case_matches_closed_form():
    # with skew 0 and kurt 3 the factor is sqrt(1 + SR^2 / 2)
    from scipy.stats import norm

    sr, t = 0.1, 253
    expected = norm.cdf(sr * np.sqrt(t - 1) / np.sqrt(1 + sr**2 / 2))
    assert probabilistic_sharpe_ratio(sr, 0.0, t) == pytest.approx(expected)


def test_fat_tails_and_negative_skew_lower_psr():
    base = probabilistic_sharpe_ratio(0.1, 0.0, 500, skew=0.0, kurt=3.0)
    assert probabilistic_sharpe_ratio(0.1, 0.0, 500, skew=-1.0, kurt=3.0) < base
    assert probabilistic_sharpe_ratio(0.1, 0.0, 500, skew=0.0, kurt=8.0) < base


def test_expected_max_sharpe_grows_with_trials_and_dispersion():
    assert expected_max_sharpe(1, 1.0) == 0.0
    e10, e100, e1000 = (expected_max_sharpe(n, 1.0) for n in (10, 100, 1000))
    assert 0 < e10 < e100 < e1000
    assert expected_max_sharpe(100, 4.0) == pytest.approx(2 * e100)
    # bounded by the crude upper bound sqrt(2 ln N)
    assert e1000 < np.sqrt(2 * np.log(1000))


def test_expected_max_sharpe_against_bailey_lopez_de_prado_example():
    # Bailey & Lopez de Prado (2014), section "A numerical example": N = 100,
    # V[SR] = 0.5 -> E[max SR] ~ 1.77 (their annual units).
    assert expected_max_sharpe(100, 0.5) == pytest.approx(1.77, abs=0.02)


def test_dsr_with_single_trial_equals_psr_vs_zero():
    dsr, sr_star = deflated_sharpe_ratio(0.08, 0.01, 1, 1000)
    assert sr_star == 0.0
    assert dsr == pytest.approx(probabilistic_sharpe_ratio(0.08, 0.0, 1000))


def test_dsr_of_the_luckiest_noise_trial_is_low():
    rng = np.random.default_rng(0)
    rejections = 0
    for _ in range(50):
        x = rng.standard_normal((2000, 200)) * 0.01
        sr = x.mean(0) / x.std(0, ddof=1)
        dsr, _ = deflated_sharpe_ratio(sr.max(), sr.var(ddof=1), 200, 2000)
        rejections += dsr < 0.95
    # the best of 200 noise strategies should almost never be certified
    assert rejections >= 45


def test_min_track_record_length_is_consistent_with_psr():
    sr, sr_star = 0.08, 0.02
    t = min_track_record_length(sr, sr_star, alpha=0.05)
    assert probabilistic_sharpe_ratio(sr, sr_star, int(np.ceil(t))) >= 0.95 - 1e-6
    assert probabilistic_sharpe_ratio(sr, sr_star, int(np.floor(t)) - 1) < 0.95
    assert min_track_record_length(0.01, 0.02) == np.inf


def test_min_backtest_length_below_upper_bound():
    n, target = 1000, 1.0
    assert min_backtest_length(n, target) < 2 * np.log(n) / target**2
    assert min_backtest_length(10, 1.0) < min_backtest_length(1000, 1.0)


def test_effective_trials_bounds():
    rng = np.random.default_rng(1)
    indep = rng.standard_normal((5000, 20))
    assert effective_number_of_trials(indep) == pytest.approx(20, rel=0.05)
    common = rng.standard_normal((5000, 1))
    clone = np.repeat(common, 20, axis=1) + 1e-6 * rng.standard_normal((5000, 20))
    assert effective_number_of_trials(clone) == pytest.approx(1, abs=0.01)
