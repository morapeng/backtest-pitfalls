"""
Moving-average crossover signal + a vectorised parameter grid, for the overfitting demo (Day 4).

`ma_crossover` is the strategy: long when the fast moving average is above the
slow one, otherwise flat (or short). It only uses prices up to the close of day t,
so it is look-ahead free when fed to `run_backtest(..., lag=1)`.

`grid_sharpe` evaluates every (fast, slow) pair at once with numpy. It applies
exactly the same rules as `backtest.run_backtest` (lag=1, costs on |Δposition|)
and exists only for speed: the permutation test re-runs the whole grid hundreds
of times. The notebook checks it against the engine before using it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def ma_crossover(prices: pd.Series, fast: int, slow: int, short: bool = False) -> pd.Series:
    """+1 when MA(fast) > MA(slow), else 0 (or -1 if short=True). 0 during warm-up."""
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    pos = pd.Series(np.where(f > s, 1.0, -1.0 if short else 0.0), index=prices.index)
    pos[s.isna()] = 0.0
    return pos


def _rolling_means(p: np.ndarray, windows) -> dict[int, np.ndarray]:
    c = np.r_[0.0, np.cumsum(p)]
    out = {}
    for w in windows:
        m = np.full(len(p), np.nan)
        m[w - 1:] = (c[w:] - c[:-w]) / w
        out[w] = m
    return out


def grid_sharpe(
    prices: pd.Series,
    fasts,
    slows,
    cost_bps: float = 0.0,
    short: bool = False,
    eval_slice: slice | None = None,
) -> pd.DataFrame:
    """Sharpe of every (fast, slow) crossover, as a fast x slow DataFrame (NaN where fast >= slow).

    Signals are computed on the full `prices` history (so MAs are warmed up),
    then the backtest is run only on the dates in `eval_slice`, exactly as
    `run_backtest(prices[eval_slice], signal[eval_slice], cost_bps, lag=1)` would.
    """
    p = prices.to_numpy(dtype=float)
    idx = np.arange(len(p))
    sel = idx[eval_slice] if eval_slice is not None else idx
    p_sel = p[sel]
    ret = np.r_[0.0, p_sel[1:] / p_sel[:-1] - 1.0]

    ma = _rolling_means(p, sorted(set(fasts) | set(slows)))
    out = pd.DataFrame(np.nan, index=pd.Index(fasts, name="fast"), columns=pd.Index(slows, name="slow"))
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            mf, ms = ma[f][sel], ma[s][sel]
            pos = np.where(mf > ms, 1.0, -1.0 if short else 0.0)
            pos[np.isnan(ms)] = 0.0
            eff = np.r_[0.0, pos[:-1]]
            trades = np.abs(np.diff(eff, prepend=0.0))
            net = eff * ret - trades * cost_bps / 1e4
            sd = net.std(ddof=1)
            out.loc[f, s] = 0.0 if sd == 0 else np.sqrt(TRADING_DAYS) * net.mean() / sd
    return out
