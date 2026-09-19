# backtest-overfitting

**Is the best backtest in a parameter search real, or just the luckiest of N?**
A small, tested Python implementation of the selection-bias corrections proposed by
Bailey and López de Prado, applied end to end to a deliberately ordinary strategy zoo
on SPY (1996–2026).

[![tests](https://github.com/iqueipopg/backtest-overfitting/actions/workflows/tests.yml/badge.svg)](https://github.com/iqueipopg/backtest-overfitting/actions/workflows/tests.yml)

- **Probabilistic Sharpe Ratio (PSR)**: probability that the true Sharpe exceeds a benchmark, adjusted for skewness, kurtosis and sample length.
- **Deflated Sharpe Ratio (DSR)**: PSR against the Sharpe you would expect from the *best of N noise strategies*, given how many trials were run and how dispersed they were.
- **Minimum track record length** and **minimum backtest length**: how much data would be needed before the winner could be certified.
- **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric Cross-Validation: in what fraction of the 12,870 in-sample/out-of-sample partitions does the in-sample winner land in the bottom half out of sample?
- A **strategy zoo** (SMA crossovers and time-series momentum, long/short, no look-ahead) that generates 147 correlated trials, plus an estimate of how many *independent* trials that really is.

Everything runs offline in about ten seconds: `data/SPY.csv` ships with the repo.

## Results on SPY, daily, 1995-12-29 to 2026-09-18 (7,730 observations)

| | no costs | 5 bp per unit turnover |
|---|---|---|
| Trials (raw N / effective N) | 147 / 2.45 | 147 / 2.45 |
| Best trial (in-sample) | TSMOM(180) | MA(50,250) |
| Best annual Sharpe | 0.52 | 0.51 |
| Median trial Sharpe | 0.20 | 0.17 |
| Buy-and-hold Sharpe | 0.61 | 0.61 |
| PSR vs. zero (naive) | **0.998** | **0.998** |
| E[max Sharpe] of N noise trials, raw N | 0.49 | 0.56 |
| **DSR, raw N** | **0.58** | **0.40** |
| E[max Sharpe] of N noise trials, effective N | 0.10 | 0.11 |
| DSR, effective N | 0.99 | 0.99 |
| Min. track record to certify, raw N | 2,281 years | never |
| PBO (CSCV, S = 16) | 0.05 | 0.04 |
| P[out-of-sample loss] | 0.004 | 0.003 |
| OOS-on-IS Sharpe slope (winner) | −0.71 | −0.70 |

What the numbers say:

1. **The naive test is fooled.** Taken on its own, the best trial has a PSR of 0.998: a 30-year daily Sharpe of 0.52 is "significant" at any conventional level. Nobody looking only at that number would reject it.
2. **Deflation kills it.** Once you account for having tried 147 configurations, the Sharpe you would expect from the luckiest *zero-skill* trial is 0.49 (0.56 with costs). The winner's 0.52 is indistinguishable from that: DSR = 0.58, well below the 0.95 threshold. Certifying it against the raw-N null would take about 2,300 years of data.
3. **How you count trials decides the verdict.** The 147 trials are variations on one idea (trend following on one asset), and the participation ratio of their correlation matrix says they amount to roughly 2.5 independent bets. Against that lenient null the DSR is 0.99. The honest answer lies between the two, which is exactly why Bailey and López de Prado insist that the effective number of trials be estimated and reported rather than assumed.
4. **PBO is low, and that is consistent, not contradictory.** CSCV asks whether the in-sample winner keeps ranking well *relative to the other trials* out of sample. It does (PBO = 0.05): long-lookback trend rules beat short-lookback ones in almost every partition, because the zoo has one dominant factor. PBO detects overfitting *within* a search; it cannot tell you the whole family is unremarkable.
5. **In-sample performance still does not carry over.** The slope of out-of-sample on in-sample Sharpe for the winner is −0.71: the partitions where the winner looked best in sample are the ones where it did worst out of sample.
6. **None of the 147 timing rules beat buy and hold in sample.** Long/short SPY at Sharpe 0.52 versus long-only at 0.61, before costs.

<p align="center">
  <img src="figures/trial_sharpes.png" width="48%">
  <img src="figures/cscv_degradation.png" width="40%">
</p>
<p align="center">
  <img src="figures/sharpe_heatmap.png" width="48%">
  <img src="figures/cscv_logits.png" width="48%">
</p>

## How it works

All Sharpe ratios are computed per period (daily) with `T` observations; annualisation is only for display.

**PSR** (Bailey & López de Prado, 2012). With `γ₃` skewness and `γ₄` (non-excess) kurtosis of the strategy's returns,

```
PSR(SR*) = Φ[ (SR − SR*) · √(T − 1) / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) ]
```

**Expected maximum Sharpe of N noise trials** (Bailey & López de Prado, 2014). With `V` the variance of the Sharpe estimates across trials and `γ` the Euler–Mascheroni constant,

```
E[max SR] ≈ √V · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
```

**DSR** = PSR of the best trial with `SR* = E[max SR]`. **MinTRL** inverts PSR for `T`. **MinBTL** (Bailey, Borwein, López de Prado & Zhu, 2014) is the number of years of backtest needed so that `E[max SR]` of N noise trials stays below a target annual Sharpe; it is bounded by `2·ln N / SR²`.

**Effective N.** Trials from one family are highly correlated. `bto.effective_number_of_trials` uses the participation ratio of the eigenvalues of the trial correlation matrix, `(Σλ)² / Σλ²`, which is N for independent trials and 1 for clones. The paper recommends clustering; this is a cheap deterministic proxy that gives the lenient end of the deflation. Both ends are reported.

**CSCV / PBO** (Bailey, Borwein, López de Prado & Zhu, 2017). The `T × N` return matrix is cut into `S = 16` blocks. For each of the `C(16, 8) = 12,870` ways to choose 8 blocks as in-sample, the in-sample best trial is selected and its relative rank `ω` among the N trials out of sample is recorded; `λ = ln(ω / (1 − ω))` and `PBO = P[λ ≤ 0]`. The implementation only needs per-block sums and sums of squares, so the full sweep is two matrix products whatever the length of the sample.

**Strategy zoo.** SMA crossovers over a 12 × 12 grid of (fast, slow) windows with fast < slow (133 trials) plus 14 time-series momentum look-backs. Positions are +1 / −1 so the equity premium does not flatter every configuration; signals are lagged one day; all trials are evaluated on the same dates (warm-up aligned to the longest window); an optional one-way cost per unit turnover.

## Run it

```bash
pip install -r requirements.txt
pytest -q                       # 20 tests, no network needed
python -m bto                   # SPY, no costs: results/summary.json + figures/
python -m bto --cost-bps 5      # with transaction costs
python -m bto --ticker QQQ --refresh   # any yfinance ticker; downloads and caches data/QQQ.csv
```

Use the pieces directly:

```python
import numpy as np
from bto import build_trials, deflated_sharpe_ratio, cscv, effective_number_of_trials, sharpe_ratio
from bto.data import load_prices

prices = load_prices("SPY")
X, trials = build_trials(prices)                  # T x N DataFrame of daily returns
sr = np.array([sharpe_ratio(X[c]) for c in X])    # per-period Sharpe of every trial
dsr, sr_star = deflated_sharpe_ratio(sr.max(), sr.var(ddof=1), len(sr), len(X))
pbo = cscv(X.to_numpy(), n_splits=16).pbo
```

## Tests

`tests/` checks the formulas against closed forms and against the numerical example in the DSR paper (N = 100, V = 0.5 gives E[max SR] ≈ 1.77), verifies that the best of 200 pure-noise trials is essentially never certified by the DSR, that PBO is about 0.5 on noise and near 0 when one trial has genuine skill, that strategy signals are lagged (no look-ahead), and that the sums-based CSCV Sharpe matches a direct computation.

## Limitations

- One asset, one family of signals. The point is the methodology, not the zoo.
- The effective-N estimate is a proxy; the paper's clustering approach would give a different, and probably larger, number.
- Sharpe ratios assume iid returns for annualisation and for the PSR standard error. Autocorrelation in strategy returns would widen the bands further.
- Transaction costs are a flat charge per unit turnover; no slippage or borrowing cost on the short leg.

## References

- Bailey, D. H. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2).
- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*, 40(5).
- Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the AMS*, 61(5).
- Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4).
- Harvey, C. R. & Liu, Y. (2015). Backtesting. *Journal of Portfolio Management*, 42(1).

## License

MIT.
