"""
Synthetic multi-stock universe with bankruptcies, for the survivorship-bias demo (Day 3).

Why synthetic: free data sources (yfinance included) mostly don't carry the
price history of companies that no longer exist — which is exactly the data
a survivorship-bias demo needs. A generated universe lets us know the ground
truth: who delisted, when, and what their "true" expected return was.

Model (daily, log returns):
  r_i,t = beta_i * market_t + alpha_i/252 + idio_vol_i/sqrt(252) * eps_i,t
A stock is delisted ("goes bankrupt") the first day its price closes below
`delist_level` x its starting price — a stylised version of an exchange's
minimum-price listing rule. On that day holders are paid `recovery` x the
closing price, after which the stock no longer trades (NaN).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Universe:
    prices: pd.DataFrame        # dates x tickers; final settled price on delist day, NaN afterwards
    delist_date: pd.Series      # ticker -> delisting date (NaT for survivors)
    params: pd.DataFrame        # ticker -> true alpha, beta, idio_vol used to simulate

    @property
    def survivors(self) -> list[str]:
        return self.delist_date.index[self.delist_date.isna()].tolist()

    def alive_count(self) -> pd.Series:
        """Number of stocks still listed at each date."""
        dead = self.delist_date.dropna()
        n_dead = pd.Series(1, index=dead.values).groupby(level=0).sum()
        n_dead = n_dead.reindex(self.prices.index, fill_value=0).cumsum()
        return len(self.delist_date) - n_dead


def make_synthetic_universe(
    n_stocks: int = 200,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
    market_mu: float = 0.07,
    market_vol: float = 0.16,
    alpha_sd: float = 0.04,
    idio_vol_range: tuple[float, float] = (0.20, 0.55),
    delist_level: float = 0.10,
    recovery: float = 0.0,
) -> Universe:
    """Generate a reproducible universe where some stocks go bust mid-sample."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end, name="Date")
    n = len(dates)
    tickers = [f"S{i:03d}" for i in range(n_stocks)]

    alpha = rng.normal(0.0, alpha_sd, n_stocks)
    beta = rng.normal(1.0, 0.25, n_stocks)
    idio_vol = rng.uniform(*idio_vol_range, n_stocks)

    market = rng.normal(market_mu / TRADING_DAYS - 0.5 * market_vol**2 / TRADING_DAYS,
                        market_vol / np.sqrt(TRADING_DAYS), n)
    eps = rng.standard_normal((n, n_stocks))
    log_ret = (np.outer(market, beta)
               + (alpha - 0.5 * idio_vol**2) / TRADING_DAYS
               + eps * idio_vol / np.sqrt(TRADING_DAYS))
    log_ret[0] = 0.0
    prices = 100.0 * np.exp(np.cumsum(log_ret, axis=0))

    delist = pd.Series(pd.NaT, index=tickers, dtype="datetime64[ns]")
    for j in range(n_stocks):
        below = np.flatnonzero(prices[:, j] < delist_level * prices[0, j])
        if below.size:
            k = below[0]
            prices[k, j] *= recovery
            prices[k + 1:, j] = np.nan
            delist.iloc[j] = dates[k]

    return Universe(
        prices=pd.DataFrame(prices, index=dates, columns=tickers),
        delist_date=delist,
        params=pd.DataFrame({"alpha": alpha, "beta": beta, "idio_vol": idio_vol}, index=tickers),
    )


def equal_weight_buy_and_hold(prices: pd.DataFrame) -> pd.Series:
    """NAV of a portfolio that puts 1/N in each column on day 0 and never trades.

    A delisted stock's settled value is carried forward as cash (forward-fill),
    so a bankruptcy with zero recovery simply leaves a hole in the portfolio.
    """
    shares = (1.0 / prices.shape[1]) / prices.iloc[0]
    return (prices.ffill() * shares).sum(axis=1).rename("NAV")
