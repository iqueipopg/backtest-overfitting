"""bto: backtest overfitting toolkit.

Implements the Probabilistic Sharpe Ratio (PSR), the Deflated Sharpe Ratio
(DSR), minimum track-record and backtest lengths, and the Probability of
Backtest Overfitting (PBO) via Combinatorially Symmetric Cross-Validation
(CSCV), together with a small strategy zoo to generate correlated trials.

References
----------
Bailey, D. H. & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier.
    Journal of Risk, 15(2).
Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
    Journal of Portfolio Management, 40(5).
Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2014).
    Pseudo-Mathematics and Financial Charlatanism. Notices of the AMS, 61(5).
Bailey, D. H., Borwein, J. M., Lopez de Prado, M. & Zhu, Q. J. (2017).
    The Probability of Backtest Overfitting. Journal of Computational Finance, 20(4).
"""

from .audit import audit_trials
from .clusters import cluster_trials
from .cscv import CSCVResult, cscv
from .holdout import HoldoutResult, holdout
from .metrics import annualize_sharpe, higher_moments, sharpe_ratio
from .psr import (
    deflated_sharpe_ratio,
    effective_number_of_trials,
    expected_max_sharpe,
    min_backtest_length,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from .simulation import simulate_max_sharpe
from .strategies import build_trials, ma_crossover, ts_momentum

__all__ = [
    "annualize_sharpe",
    "sharpe_ratio",
    "higher_moments",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_track_record_length",
    "min_backtest_length",
    "effective_number_of_trials",
    "cscv",
    "CSCVResult",
    "ma_crossover",
    "ts_momentum",
    "build_trials",
    "cluster_trials",
    "holdout",
    "HoldoutResult",
    "simulate_max_sharpe",
    "audit_trials",
]

__version__ = "0.2.0"
