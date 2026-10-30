"""
Minimal, honest backtest engine.

Core contract (keep this honest — it's the whole point of the project):
  position[t] is the weight you DECIDE at the close of day t.
  It earns the return of day t+1. So strategy_return[t+1] = position[t] * return[t+1].
  That one `.shift(1)` is the line that separates an honest backtest from a
  look-ahead-biased one. notebooks/02_lookahead.ipynb shows what happens
  when you "forget" it (lag=0).

The engine is single-asset by design. Portfolios are built elsewhere
(src/universe.py) and scored with the same metric functions defined here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series          # cumulative net value, starts at 1.0
    returns: pd.Series         # per-period strategy returns (after costs)
    sharpe: float
    max_drawdown: float        # negative number, e.g. -0.32
    cagr: float
    turnover: float            # avg absolute change in position per period

    def summary(self) -> pd.Series:
        return pd.Series({
            "CAGR": self.cagr,
            "Sharpe": self.sharpe,
            "MaxDrawdown": self.max_drawdown,
            "Turnover": self.turnover,
            "FinalEquity": float(self.equity.iloc[-1]),
        })


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / sd)


def run_backtest(
    prices: pd.Series,
    position: pd.Series,
    cost_bps: float = 0.0,
    lag: int = 1,
) -> BacktestResult:
    """Run a single-asset backtest.

    Parameters
    ----------
    prices   : price series (adjusted close).
    position : target weight per day, DECIDED using info up to that day's close.
               Typically in {-1, 0, 1} or a continuous weight.
    cost_bps : one-way transaction cost in basis points, charged on |Δposition|.
    lag      : how many days between deciding a position and earning its return.
               lag=1 is the correct, honest default. lag=0 is the look-ahead
               bug reproduced in notebooks/02_lookahead.ipynb.
    """
    prices, position = prices.align(position, join="inner")
    asset_ret = prices.pct_change().fillna(0.0)

    # the critical line: today's decision earns tomorrow's return
    effective_pos = position.shift(lag).fillna(0.0)
    gross = effective_pos * asset_ret

    # costs on turnover
    trades = effective_pos.diff().abs().fillna(effective_pos.abs())
    costs = trades * (cost_bps / 1e4)
    net = gross - costs

    equity = (1.0 + net).cumprod()

    return BacktestResult(
        equity=equity,
        returns=net,
        sharpe=sharpe_ratio(net),
        max_drawdown=max_drawdown(equity),
        cagr=float(equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1.0),
        turnover=float(trades.mean()),
    )


if __name__ == "__main__":
    # smoke test with a trivial always-long position
    from data import load_prices

    px = load_prices("SPY", start="2020-01-01", end="2023-12-31")["Close"]
    pos = pd.Series(1.0, index=px.index)  # buy & hold
    res = run_backtest(px, pos, cost_bps=1.0)
    print(res.summary().round(4))
