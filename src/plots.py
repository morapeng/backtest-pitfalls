"""
One visual style for every before/after chart in the project (Day 5).

Colour roles are fixed across all three pitfalls so a reader learns them once:
  cheat     -> orange   (the too-good curve)
  honest    -> blue     (what you'd actually get)
  reference -> neutral grey, dashed (buy & hold, where there is one)
Values come from a pre-validated categorical palette (slots 1-2 + a neutral),
with separate steps for a light and a dark background, so a figure can be
exported for either a white PS page or a dark slide.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pitfalls import PitfallResult

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "text2": "#52514e", "muted": "#8a8984",
        "grid": "#e4e3df", "axis": "#b9b8b2",
        "cheat": "#eb6834", "honest": "#2a78d6", "reference": "#7d7c77",
    },
    "dark": {
        "surface": "#1a1a19", "text": "#f0efec", "text2": "#c3c2b7", "muted": "#8f8e86",
        "grid": "#2f2f2d", "axis": "#4a4a46",
        "cheat": "#d95926", "honest": "#3987e5", "reference": "#9a9990",
    },
}

FIG_SIZE = (10, 5.2)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def _log_axis(ax, lo: float, hi: float) -> None:
    ax.set_yscale("log")
    if hi / lo < 100:  # small range: plain ticks (0.5, 0.75, 1, 1.5 ...) instead of 10^0
        nice = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50])
        ticks = nice[(nice >= lo * 0.95) & (nice <= hi * 1.05)]
        ticks = ticks[::int(np.ceil(len(ticks) / 7))]
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def _end_labels(ax, items, t, min_gap=0.075) -> None:
    """Direct labels at the right end of each line; nudged apart so they never overlap."""
    to_axes = ax.transAxes.inverted()
    x_end = ax.get_xlim()[1]
    placed = []
    for text, y, color in items:
        _, ya = to_axes.transform(ax.transData.transform((x_end, y)))
        placed.append([ya, text, y, color])
    placed.sort()
    for i in range(1, len(placed)):
        placed[i][0] = max(placed[i][0], placed[i - 1][0] + min_gap)
    for ya, text, y, color in placed:
        ax.plot([x_end], [y], marker="o", ms=6, color=color, mec=t["surface"], mew=1.5,
                clip_on=False, zorder=5)
        ax.annotate(text, xy=(1.0, ya), xycoords="axes fraction", xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=t["text2"])


def plot_pitfall(r: PitfallResult, theme: str = "light", ax=None):
    """Before/after equity chart for one pitfall, in the shared house style."""
    t = THEMES[theme]
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig = ax.figure
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    m = r.metrics()
    series = [(r.cheat, r.cheat_label, t["cheat"], "-", 2.0),
              (r.honest, r.honest_label, t["honest"], "-", 2.0)]
    if r.reference is not None:
        series.append((r.reference, r.reference_label, t["reference"], (0, (4, 3)), 1.4))

    # the illusion: area between what was reported and what was achievable
    ax.fill_between(r.cheat.index, r.honest, r.cheat, color=t["cheat"], alpha=0.10, lw=0, zorder=1)
    for eq, label, color, ls, lw in series:
        ax.plot(eq, color=color, ls=ls, lw=lw, zorder=3,
                label=f"{label}  ·  Sharpe {m.loc[label, 'Sharpe']:.2f}")
    ax.axhline(1.0, color=t["axis"], lw=0.8, zorder=2)

    all_eq = pd.concat([s[0] for s in series])
    _log_axis(ax, all_eq.min(), all_eq.max())
    ax.set_xlim(r.cheat.index[0], r.cheat.index[-1])

    # recessive axes and grid
    ax.grid(color=t["grid"], lw=0.8, which="major")
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["text2"], labelsize=9)
    ax.set_ylabel("Equity (log scale, start = 1)", color=t["text2"], fontsize=9.5)

    ax.set_title(r.title, loc="left", fontsize=14, fontweight="bold", color=t["text"], pad=24)
    ax.text(0, 1.02, r.subtitle, transform=ax.transAxes, fontsize=9.5, color=t["text2"])

    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text"])

    _end_labels(ax, [(f"Sharpe {m.loc[lbl, 'Sharpe']:.2f}", eq.iloc[-1], c)
                     for eq, lbl, c, _, _ in series], t)
    if r.notes:
        ax.text(0, -0.13, "   ".join(r.notes), transform=ax.transAxes, fontsize=8, color=t["muted"])
    fig.subplots_adjust(left=0.08, right=0.86, top=0.86, bottom=0.15)
    return fig


def format_summary(df: pd.DataFrame):
    """Styled version of pitfalls.summary_table for notebooks / Streamlit."""
    fmt = {}
    for metric, col in df.columns:
        if metric == "Sharpe":
            fmt[(metric, col)] = "{:+.2f}" if col == "Gap" else "{:.2f}"
        else:
            fmt[(metric, col)] = "{:+.1%}" if col == "Gap" else "{:.1%}"
    gap_cols = [c for c in df.columns if c[1] == "Gap"]
    return (df.style.format(fmt)
            .set_properties(subset=gap_cols, **{"font-weight": "bold"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}]))


def save_all(results: list[PitfallResult], themes=("light", "dark"), dpi: int = 200) -> list[Path]:
    """Write figures/<key>_<theme>.png for every pitfall and theme."""
    FIGURES_DIR.mkdir(exist_ok=True)
    paths = []
    for r in results:
        for theme in themes:
            fig = plot_pitfall(r, theme=theme)
            p = FIGURES_DIR / f"{r.key}_{theme}.png"
            fig.savefig(p, dpi=dpi, facecolor=fig.get_facecolor())
            plt.close(fig)
            paths.append(p)
    return paths
