import numpy as np
import pytest

from bto import cscv


def test_pbo_on_pure_noise_is_about_one_half():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((4000, 100)) * 0.01
    res = cscv(x, n_splits=10)
    assert res.n_combinations == 252
    assert 0.35 < res.pbo < 0.65
    assert res.logits.shape == (252,)


def test_pbo_is_low_when_one_trial_has_real_skill():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4000, 50)) * 0.01
    x[:, 3] += 0.0015  # annual Sharpe ~ 2.4: dominant in every partition
    res = cscv(x, n_splits=10)
    assert res.pbo < 0.05
    assert np.all(res.best_index == 3)
    assert res.prob_oos_loss < 0.05


def test_sums_based_sharpe_matches_direct_computation():
    from bto.cscv import _sharpe_from_sums

    rng = np.random.default_rng(3)
    x = rng.standard_normal((999, 4))
    s, q, n = x.sum(0)[None, :], (x**2).sum(0)[None, :], np.array([999.0])
    direct = x.mean(0) / x.std(0, ddof=1)
    assert _sharpe_from_sums(s, q, n)[0] == pytest.approx(direct)


def test_input_validation():
    x = np.zeros((100, 3))
    with pytest.raises(ValueError):
        cscv(x, n_splits=7)
    with pytest.raises(ValueError):
        cscv(x[:, :1], n_splits=4)
    with pytest.raises(ValueError):
        cscv(np.zeros((10, 3)), n_splits=16)
