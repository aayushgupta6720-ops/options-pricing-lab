"""Fit Heston (per day) and SABR (per day and expiry) to the stored chains and save the results.

    python -m scripts.calibrate --data-dir data                  # every day not fitted yet
    python -m scripts.calibrate --data-dir data --force --workers 6   # refit the whole history

Days already in models/heston.parquet are skipped unless --force. Days are fitted in date order
within runs of RUN_LENGTH days, each Heston fit warm-started from the previous day's (the starting
points also include fixed guesses, and the best fit wins, so a bad previous day doesn't stick).
Runs can go to separate processes with --workers. A day that raises is reported and skipped; the
script then exits with status 1, after saving everything else.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from optlab.calibration import fit_heston, fit_sabr_chain
from optlab.market import store
from optlab.models.heston import HestonParams

RUN_LENGTH = 40  # trading days per unit of work


def heston_row(day: date, underlying: str, fit) -> dict:
    row = {"trade_date": day, "underlying": underlying, "fitted": fit is not None}
    if fit is None:
        return row
    p = fit.params
    return {
        **row,
        "v0": p.v0,
        "kappa": p.kappa,
        "theta": p.theta,
        "xi": p.xi,
        "rho": p.rho,
        "feller_ratio": p.feller_ratio,
        "rmse": fit.rmse,
        "n_quotes": fit.n_quotes,
        "n_expiries": fit.n_expiries,
        "cost": fit.cost,
    }


def sabr_rows(day: date, underlying: str, fits) -> list[dict]:
    return [
        {
            "trade_date": day,
            "underlying": underlying,
            "expiry": f.expiry,
            "T": f.T,
            "forward": f.forward,
            "alpha": f.params.alpha,
            "rho": f.params.rho,
            "nu": f.params.nu,
            "beta": f.params.beta,
            "rmse": f.rmse,
            "n_quotes": f.n_quotes,
        }
        for f in fits
    ]


def params_from(row) -> HestonParams | None:
    if row is None or not row.get("fitted", False):
        return None
    try:
        return HestonParams(row["v0"], row["kappa"], row["theta"], row["xi"], row["rho"])
    except (ValueError, KeyError):
        return None


def fit_run(job) -> tuple[list[dict], list[dict], list[str]]:
    """Fit a run of consecutive days for one underlying. Runs in a worker process."""
    root, underlying, days, previous_row = job
    previous = params_from(previous_row)
    heston, sabr, failures = [], [], []
    months: dict[str, pd.DataFrame] = {}
    for day in days:
        month = f"{day:%Y-%m}"
        if month not in months:
            path = store.chain_path(root, underlying, month)
            months[month] = (
                store.as_dates(pd.read_parquet(path), "trade_date", "expiry") if path.exists() else None
            )
        chains = months[month]
        chain = chains[chains["trade_date"] == day] if chains is not None else pd.DataFrame()
        try:
            fit = fit_heston(chain, previous) if len(chain) else None
            sabr += sabr_rows(day, underlying, fit_sabr_chain(chain) if len(chain) else [])
        except Exception as error:  # report and keep going
            failures.append(f"{underlying} {day}: {type(error).__name__}: {error}")
            continue
        heston.append(heston_row(day, underlying, fit))
        if fit is not None:
            previous = fit.params
    return heston, sabr, failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--since", type=date.fromisoformat)
    parser.add_argument("--force", action="store_true", help="refit days that already have a fit")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    summary = store.read_summary(args.data_dir)
    if "n_quotes" in summary:
        summary = summary[summary["n_quotes"] > 0]
    if args.since:
        summary = summary[summary["trade_date"] >= args.since]
    existing = store.read_table(args.data_dir, store.HESTON_FITS)
    done = set() if args.force else set(zip(existing["trade_date"], existing["underlying"], strict=True))

    jobs = []
    for underlying, rows in summary.groupby("underlying"):
        pending = sorted(d for d in rows["trade_date"] if (d, underlying) not in done)
        fitted_before = existing[existing["underlying"] == underlying].sort_values("trade_date")
        for i in range(0, len(pending), RUN_LENGTH):
            days = pending[i : i + RUN_LENGTH]
            earlier = fitted_before[fitted_before["trade_date"] < days[0]]
            previous = earlier.iloc[-1].to_dict() if len(earlier) else None
            jobs.append((args.data_dir, underlying, days, previous))
    total = sum(len(j[2]) for j in jobs)
    if not total:
        print("Nothing to fit.")
        return 0

    print(f"Fitting {total} underlying-days in {len(jobs)} run(s) with {args.workers} worker(s)...")
    started, fitted, failures = time.perf_counter(), 0, []
    # ProcessPoolExecutor rather than multiprocessing.Pool: if a worker process dies, it raises
    # BrokenProcessPool instead of quietly replacing the worker and waiting forever.
    pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
    results = (
        (f.result() for f in as_completed([pool.submit(fit_run, job) for job in jobs]))
        if pool
        else map(fit_run, jobs)
    )
    for heston, sabr, failed in results:
        days = {(r["trade_date"], r["underlying"]) for r in heston}
        store.replace_days(args.data_dir, store.HESTON_FITS, pd.DataFrame(heston), days)
        store.replace_days(args.data_dir, store.SABR_FITS, pd.DataFrame(sabr), days)
        fitted += len(heston)
        failures += failed
        rmse = [r["rmse"] for r in heston if r["fitted"]]
        summary_text = f"median Heston RMSE {100 * np.median(rmse):.2f} pts" if rmse else "no Heston fits"
        print(f"  {fitted}/{total} done ({time.perf_counter() - started:.0f}s); last run: {summary_text}")
    if pool:
        pool.shutdown()

    print(f"Fitted {fitted} underlying-day(s).")
    for failure in failures:
        print("FAILED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
