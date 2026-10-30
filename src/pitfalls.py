"""
The three pitfalls as data: one function per pitfall returning cheat / honest
equity curves on the same dates, plus a summary table across all three (Day 5).

Each function reproduces the before/after comparison from its notebook
(02_lookahead, 03_survivorship, 04_overfitting) with the same data, parameters
and engine, so the summary notebook and the Streamlit app stay in sync with
the detailed write-ups.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import TRADING_DAYS, max_drawdown, run_backtest, sharpe_ratio
from data import load_prices
from grid import grid_sharpe, ma_crossover
from universe import equal_weight_buy_and_hold, make_synthetic_universe

COST_BPS = 1.0


@dataclass
class PitfallResult:
    key: str
    title: str
    subtitle: str              # what differs between the two curves
    cheat: pd.Series           # equity, starts at 1
    honest: pd.Series
    cheat_label: str
    honest_label: str
    reference: pd.Series | None = None
    reference_label: str | None = None
    notes: list[str] = field(default_factory=list)

    def metrics(self) -> pd.DataFrame:
        curves = {self.cheat_label: self.cheat, self.honest_label: self.honest}
        if self.reference is not None:
            curves[self.reference_label] = self.reference
        return pd.DataFrame({k: equity_stats(v) for k, v in curves.items()}).T


def equity_stats(equity: pd.Series) -> pd.Series:
    """Sharpe / CAGR / MaxDrawdown of an equity curve, using the engine's definitions."""
    returns = equity.pct_change().fillna(equity.iloc[0] - 1.0)
    return pd.Series({
        "Sharpe": sharpe_ratio(returns),
        "CAGR": equity.iloc[-1] ** (TRADING_DAYS / len(equity)) - 1.0,
        "MaxDrawdown": max_drawdown(equity),
    })


def lookahead() -> PitfallResult:
    px = load_prices("SPY", start="2015-01-01", end="2024-12-31")["Close"]
    signal = np.sign(px.pct_change().fillna(0.0))
    hold = pd.Series(1.0, index=px.index)
    return PitfallResult(
        key="lookahead",
        title="Look-ahead bias",
        subtitle="1-day momentum on SPY, 2015–2024 · same signal, only the lag differs",
        cheat=run_backtest(px, signal, COST_BPS, lag=0).equity,
        honest=run_backtest(px, signal, COST_BPS, lag=1).equity,
        cheat_label="Cheat: lag=0 (trades on today's return)",
        honest_label="Honest: lag=1",
        reference=run_backtest(px, hold, COST_BPS, lag=1).equity,
        reference_label="Buy & hold SPY",
        notes=["Real SPY data (yfinance).", "Log scale: the cheat curve grows ~60-million-fold."],
    )


def survivorship(seed: int = 42) -> PitfallResult:
    uni = make_synthetic_universe(n_stocks=200, start="2015-01-01", end="2024-12-31", seed=seed)
    n_dead = int(uni.delist_date.notna().sum())
    return PitfallResult(
        key="survivorship",
        title="Survivorship bias",
        subtitle=f"Equal-weight buy & hold · 200 synthetic stocks, {n_dead} go bankrupt · only the stock list differs",
        cheat=equal_weight_buy_and_hold(uni.prices[uni.survivors]),
        honest=equal_weight_buy_and_hold(uni.prices),
        cheat_label=f"Cheat: survivors only ({len(uni.survivors)} stocks)",
        honest_label="Honest: full universe (200 stocks)",
        notes=["Synthetic universe (seed 42): dates are labels, not real market history."],
    )


FASTS = list(range(5, 105, 5))
SLOWS = list(range(20, 310, 10))


def walk_forward(
    px: pd.Series, fasts=FASTS, slows=SLOWS, short: bool = False,
    train_years: int = 5, first: int = 2010, last: int = 2024,
) -> tuple[pd.Series, pd.DataFrame]:
    """Each year, pick the best pair on the trailing `train_years`, trade it for that year.

    Returns the stitched out-of-sample daily returns and one row per year with
    the chosen pair and its in-sample / out-of-sample Sharpe.
    """
    years = px.index.year
    out, picks = [], []
    for y in range(first, last + 1):
        tr = np.flatnonzero((years >= y - train_years) & (years < y))
        te = np.flatnonzero(years == y)
        g = grid_sharpe(px, fasts, slows, COST_BPS, short=short, eval_slice=slice(tr[0], tr[-1] + 1)).stack()
        f, s = g.idxmax()
        # start one day early so the first test day earns the position decided at the previous close
        sl = slice(te[0] - 1, te[-1] + 1)
        r = run_backtest(px.iloc[sl], ma_crossover(px, f, s, short=short).iloc[sl], COST_BPS, lag=1).returns.iloc[1:]
        out.append(r)
        picks.append({"year": y, "fast": f, "slow": s, "in_sample": g.max(), "out_of_sample": sharpe_ratio(r)})
    return pd.concat(out), pd.DataFrame(picks).set_index("year")


def overfitting() -> PitfallResult:
    px = load_prices("SPY", start="2005-01-01", end="2024-12-31")["Close"]
    best = grid_sharpe(px, FASTS, SLOWS, COST_BPS).stack().idxmax()

    wf, _ = walk_forward(px)
    window = wf.index
    cheat_ret = run_backtest(px, ma_crossover(px, *best), COST_BPS, lag=1).returns.loc[window]
    bh_ret = run_backtest(px, pd.Series(1.0, index=px.index), COST_BPS, lag=1).returns.loc[window]
    return PitfallResult(
        key="overfitting",
        title="Overfitting / parameter mining",
        subtitle="MA trend filter on SPY, 2010–2024 · best of 499 pairs in-sample vs walk-forward out-of-sample",
        cheat=(1 + cheat_ret).cumprod(),
        honest=(1 + wf).cumprod(),
        cheat_label=f"Cheat: MA {best[0]}/{best[1]}, picked on 2005–2024",
        honest_label="Honest: walk-forward (re-picked yearly)",
        reference=(1 + bh_ret).cumprod(),
        reference_label="Buy & hold SPY",
        notes=["Real SPY data (yfinance).", "Out-of-sample window 2010–2024 for all three curves."],
    )


PITFALLS = {"lookahead": lookahead, "survivorship": survivorship, "overfitting": overfitting}


def summary_table(results: list[PitfallResult]) -> pd.DataFrame:
    """Rows = pitfalls; columns = (metric, Cheat / Honest / Gap), gap = honest − cheat."""
    rows = {}
    for r in results:
        m = r.metrics()
        cheat, honest = m.loc[r.cheat_label], m.loc[r.honest_label]
        for metric in ["Sharpe", "CAGR", "MaxDrawdown"]:
            rows.setdefault(r.title, {})
            rows[r.title][(metric, "Cheat")] = cheat[metric]
            rows[r.title][(metric, "Honest")] = honest[metric]
            rows[r.title][(metric, "Gap")] = honest[metric] - cheat[metric]
    df = pd.DataFrame(rows).T
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df
