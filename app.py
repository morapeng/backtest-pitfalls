"""
Streamlit page: pick a pitfall, see its before/after chart and metrics.

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pitfalls import PITFALLS, summary_table  # noqa: E402
from plots import format_summary, plot_pitfall  # noqa: E402

LESSONS = {
    "lookahead": (
        "The position decided at today's close is paid **today's** return, which it already "
        "knows, because the engine runs with `lag=0`. The fix is one argument: `lag=1`, so a "
        "decision taken at the close of day *t* earns the return of day *t+1*."
    ),
    "survivorship": (
        "The cheat builds its portfolio from the stocks still listed at the **end** of the sample, "
        "silently dropping every company that went bankrupt. The honest version starts from every "
        "stock listed on day one. Most of the gap comes from *which* stocks survive, not from the "
        "size of the bankruptcy losses."
    ),
    "overfitting": (
        "Every backtest in the grid is honest. The bias is in reporting the **best of 499** on the "
        "same data used to choose it. Walk-forward testing measures the selection *process* "
        "out-of-sample, and the edge over buy & hold disappears."
    ),
}

st.set_page_config(page_title="Backtest Pitfalls", layout="wide")


@st.cache_data(show_spinner="Running backtests…")
def load_results():
    return {key: build() for key, build in PITFALLS.items()}


def current_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit without st.context.theme
        return "light"


results = load_results()
theme = current_theme()

st.title("Backtest Pitfalls")
st.caption("How a few bugs manufacture a beautiful — and fake — equity curve.")

key = st.selectbox(
    "Pitfall",
    options=list(results),
    format_func=lambda k: results[k].title,
)
r = results[key]

st.subheader(r.title)
st.markdown(LESSONS[key])
st.pyplot(plot_pitfall(r, theme=theme), width="stretch")

st.markdown("**Metrics**")
st.dataframe(
    r.metrics().style.format({"Sharpe": "{:.2f}", "CAGR": "{:.1%}", "MaxDrawdown": "{:.1%}"}),
    width="stretch",
)
if r.notes:
    st.caption(" ".join(r.notes))

st.divider()
st.markdown("**All three pitfalls — cheat vs honest (gap = honest − cheat)**")
st.dataframe(format_summary(summary_table(list(results.values()))), width="stretch")
