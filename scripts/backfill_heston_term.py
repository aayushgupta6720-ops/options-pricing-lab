"""Add Heston theta(t) to the stored rough Bergomi fits, without refitting rough Bergomi.

    python -m scripts.backfill_heston_term --data-dir data --workers 8

The daily job (scripts/calibrate.py, model "rough") fits Heston with one long-run variance per
expiry next to rough Bergomi. This fills those columns in for days fitted before it did: for each
fitted day it refits both Heston variants to rough Bergomi's window and quotes, and writes the
theta(t) errors and parameters into the day, expiry and quote tables. Rows that already have
them are skipped unless --force.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from optlab import calibration
from optlab.market import store
from scripts.calibrate import _chains, same_window_heston, term_columns


def backfill_days(job) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    """theta(t) columns for some days of one underlying: (day rows, expiry rows, quote rows, failures).
    Runs in a worker process."""
    root, underlying, days = job
    chain_for = _chains(root, underlying)
    day_rows, expiry_rows, quote_rows, failures = [], [], [], []
    for day in days:
        try:
            chain = chain_for(day)
            quotes = pd.concat(
                [
                    q
                    for _, q in calibration._usable_expiries(
                        chain, calibration.ROUGH_MIN_DAYS, calibration.ROUGH_MAX_DAYS
                    )
                ]
            )
            heston, heston_tt = same_window_heston(chain)
            row, by_expiry, vols = term_columns(quotes, heston_tt)
        except Exception as error:  # report and keep going
            failures.append(f"{underlying} {day}: {type(error).__name__}: {error}")
            continue
        key = {"trade_date": day, "underlying": underlying}
        day_rows.append({**key, **row})
        expiry_rows += [{**key, "expiry": e, **values} for e, values in by_expiry.to_dict("index").items()]
        quote_rows += (
            quotes[["expiry", "strike", "option_type"]].assign(**key, heston_tt_iv=vols).to_dict("records")
        )
    return day_rows, expiry_rows, quote_rows, failures


def merge(table: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """`table` with `new`'s columns set on the rows they key; rows `new` doesn't have keep what an
    earlier backfill stored for them, if anything."""
    if new.empty:
        return table
    t, n = table.set_index(keys), new.set_index(keys)
    present = [c for c in n.columns if c in t]
    if present:
        n = pd.concat([t.loc[~t.index.isin(n.index), present], n])
    return t.drop(columns=present).join(n, how="left").reset_index()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="redo days that already have theta(t)")
    args = parser.parse_args(argv)

    fits = store.read_table(args.data_dir, store.ROUGH_FITS)
    todo = fits[fits["fitted"].astype(bool)]
    if not args.force and "heston_tt_rmse" in todo:
        todo = todo[todo["heston_tt_rmse"].isna()]
    jobs = [
        (args.data_dir, underlying, sorted(rows["trade_date"])[i : i + 20])
        for underlying, rows in todo.groupby("underlying")
        for i in range(0, len(rows), 20)
    ]
    if not jobs:
        print("Nothing to backfill.")
        return 0
    print(f"Backfilling Heston theta(t) for {len(todo)} day(s) in {len(jobs)} run(s)...")
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(backfill_days, jobs))
    day_rows = pd.DataFrame([r for result in results for r in result[0]])
    expiry_rows = pd.DataFrame([r for result in results for r in result[1]])
    quote_rows = pd.DataFrame([r for result in results for r in result[2]])
    failures = [f for result in results for f in result[3]]

    for relative, new, keys in (
        (store.ROUGH_FITS, day_rows, ["trade_date", "underlying"]),
        (store.ROUGH_EXPIRIES, expiry_rows, ["trade_date", "underlying", "expiry"]),
        (store.ROUGH_QUOTES, quote_rows, ["trade_date", "underlying", "expiry", "strike", "option_type"]),
    ):
        table = store.read_table(args.data_dir, relative)
        merged = merge(table, new, keys)
        days = set(zip(merged["trade_date"], merged["underlying"], strict=True))
        store.replace_days(args.data_dir, relative, merged, days)

    done = day_rows.dropna(subset=["heston_tt_rmse"]) if len(day_rows) else day_rows
    print(f"Done: {len(done)} day(s) with theta(t).")
    if len(done):
        both = fits.drop(
            columns=[c for c in done.columns if c in fits and c not in ("trade_date", "underlying")]
        )
        both = both.merge(done, on=["trade_date", "underlying"])
        shape = {
            c: 100 * both[c].median() for c in ("shape_rmse", "heston_shape_rmse", "heston_tt_shape_rmse")
        }
        print(
            f"Median shape error: rough {shape['shape_rmse']:.3f}, Heston {shape['heston_shape_rmse']:.3f}, "
            f"Heston theta(t) {shape['heston_tt_shape_rmse']:.3f} vol pts"
        )
    for failure in failures:
        print("FAILED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
