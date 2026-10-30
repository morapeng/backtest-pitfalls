"""
Key-number check: recompute every figure quoted in docs/writeup.md and compare it
with the value the write-up states (Day 7).

Each check is compared at the precision the write-up prints it, so a mismatch
means the write-up and the code disagree, e.g. after a data refresh or a code
change. `run_all.py` prints the table and fails if any row mismatches.

The extra statistics (train/test split, permutation test, ...) mirror the code
in the notebooks exactly, including random seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import TRADING_DAYS, run_backtest, sharpe_ratio
from data import load_prices
from grid import grid_sharpe, ma_crossover
from pitfalls import COST_BPS, FASTS, SLOWS, PitfallResult, summary_table, walk_forward
from universe import equal_weight_buy_and_hold, make_synthetic_universe

# (section, label, value as stated in docs/writeup.md, format used there)
WRITEUP = [
    ("Look-ahead", "cheat Sharpe", 13.64, "{:.2f}"),
    ("Look-ahead", "cheat CAGR", 5.043, "{:.1%}"),
    ("Look-ahead", "cheat MaxDD", -0.0002, "{:.2%}"),
    ("Look-ahead", "honest Sharpe", -0.36, "{:.2f}"),
    ("Look-ahead", "honest CAGR", -0.076, "{:.1%}"),
    ("Look-ahead", "honest MaxDD", -0.652, "{:.1%}"),
    ("Look-ahead", "buy & hold Sharpe", 0.79, "{:.2f}"),
    ("Look-ahead", "buy & hold CAGR", 0.131, "{:.1%}"),
    ("Look-ahead", "buy & hold MaxDD", -0.337, "{:.1%}"),
    ("Look-ahead", "cheat final equity ($M)", 63, "{:.0f}"),
    ("Look-ahead", "gross return == |r| every day", 1, "{:.0f}"),
    ("Look-ahead", "SPY return 2020-03-16", -0.1094, "{:.2%}"),
    ("Survivorship", "stocks delisted", 47, "{:.0f}"),
    ("Survivorship", "cheat Sharpe", 0.36, "{:.2f}"),
    ("Survivorship", "cheat CAGR", 0.0478, "{:.2%}"),
    ("Survivorship", "cheat MaxDD", -0.554, "{:.1%}"),
    ("Survivorship", "cheat final value", 1.62, "{:.2f}"),
    ("Survivorship", "honest Sharpe", 0.21, "{:.2f}"),
    ("Survivorship", "honest CAGR", 0.0211, "{:.2%}"),
    ("Survivorship", "honest MaxDD", -0.588, "{:.1%}"),
    ("Survivorship", "honest final value", 1.24, "{:.2f}"),
    ("Survivorship", "true alpha, delisted", -0.0112, "{:.2%}"),
    ("Survivorship", "true alpha, survived", 0.0018, "{:.2%}"),
    ("Survivorship", "idio vol, delisted", 0.437, "{:.1%}"),
    ("Survivorship", "idio vol, survived", 0.359, "{:.1%}"),
    ("Survivorship", "bankruptcies in 2020", 17, "{:.0f}"),
    ("Survivorship", "CAGR gap, recovery 0% (pp)", 2.68, "{:.2f}"),
    ("Survivorship", "CAGR gap, recovery 50% (pp)", 2.58, "{:.2f}"),
    ("Survivorship", "CAGR gap, recovery 100% (pp)", 2.49, "{:.2f}"),
    ("Survivorship", "100 seeds: universes with gap > 0", 100, "{:.0f}"),
    ("Survivorship", "100 seeds: mean gap (pp)", 1.52, "{:.2f}"),
    ("Survivorship", "100 seeds: median gap (pp)", 1.37, "{:.2f}"),
    ("Survivorship", "100 seeds: min gap (pp)", 0.28, "{:.2f}"),
    ("Survivorship", "100 seeds: max gap (pp)", 4.53, "{:.2f}"),
    ("Survivorship", "100 seeds: mean delisted share", 0.136, "{:.1%}"),
    ("Overfitting", "full-sample best Sharpe (MA 10/80)", 0.82, "{:.2f}"),
    ("Overfitting", "full-sample best CAGR", 0.090, "{:.1%}"),
    ("Overfitting", "full-sample best MaxDD", -0.204, "{:.1%}"),
    ("Overfitting", "buy & hold 2005-2024 Sharpe", 0.61, "{:.2f}"),
    ("Overfitting", "buy & hold 2005-2024 CAGR", 0.103, "{:.1%}"),
    ("Overfitting", "buy & hold 2005-2024 MaxDD", -0.552, "{:.1%}"),
    ("Overfitting", "train-best (45/160) Sharpe, train", 0.96, "{:.2f}"),
    ("Overfitting", "train-best (45/160) Sharpe, test", 0.71, "{:.2f}"),
    ("Overfitting", "buy & hold Sharpe, test 2015-2024", 0.79, "{:.2f}"),
    ("Overfitting", "train/test Sharpe correlation", -0.04, "{:.2f}"),
    ("Overfitting", "walk-forward mean in-sample Sharpe", 1.10, "{:.2f}"),
    ("Overfitting", "walk-forward mean out-of-sample Sharpe", 0.69, "{:.2f}"),
    ("Overfitting", "2010-2024 cheat Sharpe", 0.79, "{:.2f}"),
    ("Overfitting", "2010-2024 walk-forward Sharpe", 0.55, "{:.2f}"),
    ("Overfitting", "2010-2024 walk-forward CAGR", 0.060, "{:.1%}"),
    ("Overfitting", "2010-2024 walk-forward MaxDD", -0.246, "{:.1%}"),
    ("Overfitting", "2010-2024 buy & hold Sharpe", 0.85, "{:.2f}"),
    ("Overfitting", "2010-2024 buy & hold CAGR", 0.138, "{:.1%}"),
    ("Overfitting", "long/short full-sample best Sharpe", 0.51, "{:.2f}"),
    ("Overfitting", "long/short walk-forward Sharpe", -0.09, "{:.2f}"),
    ("Overfitting", "winner minus median Sharpe", 0.15, "{:.2f}"),
    ("Overfitting", "standard error of one Sharpe", 0.26, "{:.2f}"),
    ("Overfitting", "expected best of 500 zero-skill", 0.68, "{:.2f}"),
    ("Overfitting", "permutation p-value", 0.15, "{:.2f}"),
    ("Overfitting", "median pair minus B&H, real", 0.05, "{:+.2f}"),
    ("Overfitting", "median pair minus B&H, shuffled", -0.14, "{:+.2f}"),
    ("Summary", "Sharpe gap, look-ahead", -14.00, "{:.2f}"),
    ("Summary", "Sharpe gap, survivorship", -0.15, "{:.2f}"),
    ("Summary", "Sharpe gap, overfitting", -0.25, "{:.2f}"),
]


def _cagr(nav: pd.Series) -> float:
    return nav.iloc[-1] ** (TRADING_DAYS / len(nav)) - 1


def _lookahead(res: PitfallResult) -> dict:
    m = res.metrics()
    px = load_prices("SPY", start="2015-01-01", end="2024-12-31")["Close"]
    ret = px.pct_change().fillna(0.0)
    c, h, b = m.loc[res.cheat_label], m.loc[res.honest_label], m.loc[res.reference_label]
    return {
        "cheat Sharpe": c.Sharpe, "cheat CAGR": c.CAGR, "cheat MaxDD": c.MaxDrawdown,
        "honest Sharpe": h.Sharpe, "honest CAGR": h.CAGR, "honest MaxDD": h.MaxDrawdown,
        "buy & hold Sharpe": b.Sharpe, "buy & hold CAGR": b.CAGR, "buy & hold MaxDD": b.MaxDrawdown,
        "cheat final equity ($M)": res.cheat.iloc[-1] / 1e6,
        "gross return == |r| every day": float(np.allclose(np.sign(ret) * ret, ret.abs())),
        "SPY return 2020-03-16": ret.loc["2020-03-16"],
    }


def _survivorship(res: PitfallResult) -> dict:
    m = res.metrics()
    c, h = m.loc[res.cheat_label], m.loc[res.honest_label]
    uni = make_synthetic_universe(seed=42)
    fate = uni.params.groupby(np.where(uni.delist_date.isna(), "survived", "delisted")).mean()
    out = {
        "stocks delisted": uni.delist_date.notna().sum(),
        "cheat Sharpe": c.Sharpe, "cheat CAGR": c.CAGR, "cheat MaxDD": c.MaxDrawdown,
        "cheat final value": res.cheat.iloc[-1],
        "honest Sharpe": h.Sharpe, "honest CAGR": h.CAGR, "honest MaxDD": h.MaxDrawdown,
        "honest final value": res.honest.iloc[-1],
        "true alpha, delisted": fate.loc["delisted", "alpha"],
        "true alpha, survived": fate.loc["survived", "alpha"],
        "idio vol, delisted": fate.loc["delisted", "idio_vol"],
        "idio vol, survived": fate.loc["survived", "idio_vol"],
        "bankruptcies in 2020": (uni.delist_date.dt.year == 2020).sum(),
    }
    for rec in (0.0, 0.5, 1.0):
        u = make_synthetic_universe(seed=42, recovery=rec)
        gap = _cagr(equal_weight_buy_and_hold(u.prices[u.survivors])) - _cagr(equal_weight_buy_and_hold(u.prices))
        out[f"CAGR gap, recovery {rec:.0%} (pp)"] = 100 * gap

    gaps, dead = [], []
    for s in range(100):
        u = make_synthetic_universe(seed=s)
        gaps.append(100 * (_cagr(equal_weight_buy_and_hold(u.prices[u.survivors]))
                           - _cagr(equal_weight_buy_and_hold(u.prices))))
        dead.append(u.delist_date.notna().mean())
    gaps = np.array(gaps)
    out.update({
        "100 seeds: universes with gap > 0": (gaps > 0).sum(),
        "100 seeds: mean gap (pp)": gaps.mean(), "100 seeds: median gap (pp)": np.median(gaps),
        "100 seeds: min gap (pp)": gaps.min(), "100 seeds: max gap (pp)": gaps.max(),
        "100 seeds: mean delisted share": np.mean(dead),
    })
    return out


def _overfitting(res: PitfallResult) -> dict:
    px = load_prices("SPY", start="2005-01-01", end="2024-12-31")["Close"]
    years = px.index.year
    hold = lambda p: pd.Series(1.0, index=p.index)

    full = grid_sharpe(px, FASTS, SLOWS, COST_BPS).stack()
    best = full.idxmax()
    cheat = run_backtest(px, ma_crossover(px, *best), COST_BPS)
    bh = run_backtest(px, hold(px), COST_BPS)

    n_train = int((years < 2015).sum())
    train = grid_sharpe(px, FASTS, SLOWS, COST_BPS, eval_slice=slice(0, n_train)).stack()
    test = grid_sharpe(px, FASTS, SLOWS, COST_BPS, eval_slice=slice(n_train, None)).stack()
    b_tr = train.idxmax()
    bh_test = run_backtest(px.iloc[n_train:], hold(px.iloc[n_train:]), COST_BPS).sharpe

    _, picks = walk_forward(px)
    wf_ls, _ = walk_forward(px, short=True)
    full_ls = grid_sharpe(px, FASTS, SLOWS, COST_BPS, short=True).stack()

    m = res.metrics()
    c, h, b = m.loc[res.cheat_label], m.loc[res.honest_label], m.loc[res.reference_label]

    t_years = len(px) / TRADING_DAYS
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4000, 1000)) / np.sqrt(t_years)

    ret = px.pct_change().fillna(0.0).to_numpy()
    rng = np.random.default_rng(1)
    null_best, null_median = [], []
    for _ in range(200):
        shuffled = np.r_[0.0, rng.permutation(ret[1:])]
        p = pd.Series(px.iloc[0] * np.cumprod(1 + shuffled), index=px.index)
        g = grid_sharpe(p, FASTS, SLOWS, COST_BPS).stack()
        bh_s = sharpe_ratio(p.pct_change().fillna(0.0).iloc[1:])
        null_best.append(g.max() - bh_s)
        null_median.append(g.median() - bh_s)
    bh_real = sharpe_ratio(pd.Series(ret[1:]))

    return {
        "full-sample best Sharpe (MA 10/80)": full.max(),
        "full-sample best CAGR": cheat.cagr, "full-sample best MaxDD": cheat.max_drawdown,
        "buy & hold 2005-2024 Sharpe": bh.sharpe, "buy & hold 2005-2024 CAGR": bh.cagr,
        "buy & hold 2005-2024 MaxDD": bh.max_drawdown,
        "train-best (45/160) Sharpe, train": train[b_tr], "train-best (45/160) Sharpe, test": test[b_tr],
        "buy & hold Sharpe, test 2015-2024": bh_test,
        "train/test Sharpe correlation": train.corr(test),
        "walk-forward mean in-sample Sharpe": picks["in_sample"].mean(),
        "walk-forward mean out-of-sample Sharpe": picks["out_of_sample"].mean(),
        "2010-2024 cheat Sharpe": c.Sharpe,
        "2010-2024 walk-forward Sharpe": h.Sharpe, "2010-2024 walk-forward CAGR": h.CAGR,
        "2010-2024 walk-forward MaxDD": h.MaxDrawdown,
        "2010-2024 buy & hold Sharpe": b.Sharpe, "2010-2024 buy & hold CAGR": b.CAGR,
        "long/short full-sample best Sharpe": full_ls.max(),
        "long/short walk-forward Sharpe": sharpe_ratio(wf_ls),
        "winner minus median Sharpe": full.max() - full.median(),
        "standard error of one Sharpe": np.sqrt((1 + full.max() ** 2 / 2) / t_years),
        "expected best of 500 zero-skill": noise[:, :500].max(axis=1).mean(),
        "permutation p-value": np.mean(np.array(null_best) >= full.max() - bh_real),
        "median pair minus B&H, real": full.median() - bh_real,
        "median pair minus B&H, shuffled": np.median(null_median),
    }


def key_numbers(results: dict[str, PitfallResult]) -> pd.DataFrame:
    """One row per number in the write-up: recomputed value vs stated value, and whether they agree."""
    summary = summary_table(list(results.values()))
    computed = {
        "Look-ahead": _lookahead(results["lookahead"]),
        "Survivorship": _survivorship(results["survivorship"]),
        "Overfitting": _overfitting(results["overfitting"]),
        "Summary": {
            f"Sharpe gap, {k}": summary.loc[results[key].title, ("Sharpe", "Gap")]
            for k, key in [("look-ahead", "lookahead"), ("survivorship", "survivorship"),
                           ("overfitting", "overfitting")]
        },
    }
    rows = []
    for section, label, stated, fmt in WRITEUP:
        value = computed[section][label]
        rows.append({
            "section": section, "number": label,
            "write-up": fmt.format(stated), "recomputed": fmt.format(value),
            "ok": fmt.format(stated) == fmt.format(value),
        })
    return pd.DataFrame(rows)
