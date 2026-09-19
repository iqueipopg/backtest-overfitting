"""Price data: cached CSV first, yfinance only when asked or missing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(ticker: str = "SPY", start: str = "1995-01-01", refresh: bool = False) -> pd.Series:
    """Adjusted close series for ``ticker``.

    Reads ``data/<ticker>.csv`` when present (the repository ships SPY so the
    experiment is reproducible offline); otherwise downloads with yfinance and
    caches the result.
    """
    path = DATA_DIR / f"{ticker}.csv"
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = ticker
        return s.loc[start:].astype(float)

    import yfinance as yf  # imported lazily: tests never need it

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna().astype(float)
    close.name = ticker
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    close.to_frame("adj_close").to_csv(path)
    return close
