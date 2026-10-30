"""
Data layer: load daily OHLCV prices, cache to parquet.

Design notes:
  - yfinance fetches real, dividend-adjusted prices from Yahoo.
  - Everything is cached to data/*.parquet, so the network is hit once and
    later runs reproduce the same numbers exactly.
  - If the fetch fails (no network), a deterministic synthetic generator
    (seeded by crc32 of the ticker) keeps the pipeline running end-to-end.
    The published numbers all come from the real cached prices; the
    survivorship demo uses its own synthetic universe (src/universe.py).
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# project_root/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str, start: str, end: str) -> Path:
    key = f"{ticker}_{start}_{end}".replace(":", "-")
    return DATA_DIR / f"{key}.parquet"


def _fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Return a tidy OHLCV frame from Yahoo, or None if it fails."""
    try:
        import yfinance as yf

        df = yf.download(
            ticker, start=start, end=end,
            progress=False, auto_adjust=True,
        )
        if df is None or len(df) == 0:
            return None
        # yfinance may return a MultiIndex column frame for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        return df
    except Exception as e:  # network blocked, bad ticker, etc.
        print(f"[data] yfinance failed for {ticker}: {e!r} -> synthetic fallback")
        return None


def _synthetic(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Deterministic geometric-brownian-motion prices. Seed = crc32(ticker), stable across processes."""
    dates = pd.bdate_range(start=start, end=end, name="Date")
    n = len(dates)
    seed = zlib.crc32(ticker.encode())
    rng = np.random.default_rng(seed)

    mu, sigma = 0.08 / 252, 0.20 / np.sqrt(252)  # ~8% drift, 20% vol annualised
    rets = rng.normal(mu, sigma, n)
    close = 100 * np.exp(np.cumsum(rets))

    # build plausible OHLCV around the close
    intraday = np.abs(rng.normal(0, 0.01, n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.integers(1_000_000, 5_000_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def load_prices(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load one ticker's OHLCV frame, cached to parquet.

    Tries yfinance first, falls back to synthetic data if the fetch fails.
    """
    cache = _cache_path(ticker, start, end)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    df = _fetch_yfinance(ticker, start, end)
    if df is None:
        df = _synthetic(ticker, start, end)

    df.to_parquet(cache)
    return df


def load_universe(
    tickers: list[str],
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted Close prices (columns = tickers)."""
    closes = {t: load_prices(t, start, end)["Close"] for t in tickers}
    return pd.DataFrame(closes).sort_index()


if __name__ == "__main__":
    px = load_universe(["AAPL", "MSFT", "SPY"], start="2020-01-01", end="2020-06-30")
    print(px.tail())
    print("shape:", px.shape)
