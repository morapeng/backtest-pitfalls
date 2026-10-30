"""
One-command reproduction of the whole project.

    pip install -r requirements.txt
    python run_all.py                  # uses the cached prices in data/
    python run_all.py --refresh-data   # re-download prices from Yahoo first
    python run_all.py --skip-notebooks # numbers + figures only, faster

Steps, in order:
  1. data       load (or fetch) the SPY price histories the pitfalls use
  2. pitfalls   re-execute notebooks 02-04 in place (look-ahead, survivorship, overfitting)
  3. figures    rebuild the three before/after comparisons, export figures/*.png
  4. summary    re-execute notebook 05 and print the cheat-vs-honest summary table
  5. check      recompute every number quoted in docs/writeup.md and compare

Exits with status 1 if a notebook fails or any number disagrees with the write-up.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # no windows; figures are only written to disk

import pandas as pd  # noqa: E402

from checks import key_numbers  # noqa: E402
from data import load_prices  # noqa: E402
from pitfalls import PITFALLS, summary_table  # noqa: E402
from plots import save_all  # noqa: E402

PRICE_WINDOWS = [("SPY", "2005-01-01", "2024-12-31"), ("SPY", "2015-01-01", "2024-12-31")]
PITFALL_NOTEBOOKS = ["02_lookahead.ipynb", "03_survivorship.ipynb", "04_overfitting.ipynb"]
SUMMARY_NOTEBOOK = "05_summary.ipynb"


def step(n: int, title: str) -> None:
    print(f"\n[{n}/5] {title}", flush=True)


def execute_notebook(name: str) -> None:
    import nbformat
    from nbclient import NotebookClient

    path = ROOT / "notebooks" / name
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    print(f"      {name:<24} ok  ({time.time() - t0:.0f}s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-data", action="store_true", help="re-download prices instead of using data/ cache")
    ap.add_argument("--skip-notebooks", action="store_true", help="don't re-execute the notebooks")
    args = ap.parse_args()

    if sys.platform == "win32":  # pyzmq (used by the notebook kernel) needs the selector loop on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    t_start = time.time()

    step(1, "data")
    for ticker, start, end in PRICE_WINDOWS:
        px = load_prices(ticker, start=start, end=end, use_cache=not args.refresh_data)
        print(f"      {ticker} {start} -> {end}: {len(px)} rows "
              f"({px.index[0].date()} -> {px.index[-1].date()})", flush=True)

    step(2, "pitfalls: notebooks 02-04")
    if args.skip_notebooks:
        print("      skipped (--skip-notebooks)")
    else:
        for name in PITFALL_NOTEBOOKS:
            execute_notebook(name)

    step(3, "figures")
    results = {key: build() for key, build in PITFALLS.items()}
    for p in save_all(list(results.values())):
        print(f"      {p.relative_to(ROOT).as_posix()}")

    step(4, "summary")
    if not args.skip_notebooks:
        execute_notebook(SUMMARY_NOTEBOOK)
    summary = summary_table(list(results.values()))
    shown = summary.copy()
    for col in shown.columns:
        fmt = ("{:.2f}" if col[0] == "Sharpe" else "{:.1%}").replace("{:", "{:+" if col[1] == "Gap" else "{:")
        shown[col] = shown[col].map(fmt.format)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(shown.to_string())

    step(5, "key numbers vs docs/writeup.md")
    checks = key_numbers(results)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.max_colwidth", 60):
        print(checks.assign(ok=checks["ok"].map({True: "OK", False: "MISMATCH"})).to_string(index=False))

    n_bad = int((~checks["ok"]).sum())
    print(f"\n{len(checks) - n_bad}/{len(checks)} numbers match the write-up. "
          f"Total time {time.time() - t_start:.0f}s.")
    if n_bad:
        print("Some numbers changed: update docs/writeup.md (and src/checks.py) before publishing.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
