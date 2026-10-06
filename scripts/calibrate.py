"""Fit the models to the stored chains and save the results.

    python -m scripts.calibrate --data-dir data                        # every day not fitted yet
    python -m scripts.calibrate --data-dir data --force --workers 8    # refit the whole history
    python -m scripts.calibrate --data-dir data --models rough         # just rough Bergomi

Models:
    heston  Heston per day and SABR per day and expiry, every underlying
    rough   rough Bergomi per day (config.ROUGH_UNDERLYINGS), with Heston refitted to the same
            2-91 day window for a like-for-like comparison, per-expiry skews and per-quote vols

Days already fitted are skipped unless --force. Days are fitted in date order within runs, each fit
warm-started from the previous day's (fixed starting points are tried too, and the best fit wins,
so a bad previous day doesn't stick). Runs can go to separate processes with --workers. A day that
raises is reported and skipped; the script then exits with status 1, after saving everything else.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from optlab import calibration, config
from optlab.calibration import (
    fit_heston,
    fit_rough_bergomi,
    fit_sabr_chain,
    heston_skew,
    heston_vols,
    shape_rmse,
)
from optlab.market import store
from optlab.models.heston import HestonParams
from optlab.models.rough_bergomi import RoughBergomiParams


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


def rough_rows(day: date, underlying: str, fit, heston) -> tuple[dict, list[dict], list[dict]]:
    """The day's row, per-expiry rows and per-quote rows for a rough Bergomi fit, with the
    same-window Heston fit alongside."""
    row = {"trade_date": day, "underlying": underlying, "fitted": fit is not None}
    if fit is None:
        return row, [], []
    quotes = fit.quotes.copy()
    quotes["heston_iv"] = heston_vols(quotes, heston.params) if heston else np.nan
    rough_miss, heston_miss = quotes["model_iv"] - quotes["iv"], quotes["heston_iv"] - quotes["iv"]
    p = fit.params
    row.update(
        H=p.H,
        eta=p.eta,
        rho=p.rho,
        rmse=fit.rmse,
        shape_rmse=fit.shape_rmse,
        heston_rmse=float(np.sqrt(np.nanmean(heston_miss**2))) if heston else np.nan,
        heston_shape_rmse=shape_rmse(heston_miss, quotes["expiry"]) if heston else np.nan,
        n_quotes=fit.n_quotes,
        n_expiries=fit.n_expiries,
        cost=fit.cost,
        xi_times=list(map(float, fit.xi.times)),
        xi_values=list(map(float, fit.xi.values)),
    )

    def per_expiry(miss):
        groups = miss.groupby(quotes["expiry"].to_numpy())
        return groups.apply(lambda m: np.sqrt(np.nanmean(m**2))), groups.apply(lambda m: np.nanstd(m))

    rough_rmse, rough_shape = per_expiry(rough_miss)
    heston_rmse, heston_shape = per_expiry(heston_miss)
    counts = quotes.groupby("expiry").size()
    expiries = []
    for e in fit.expiries.itertuples():
        expiries.append(
            {
                "trade_date": day,
                "underlying": underlying,
                "expiry": e.expiry,
                "T": e.T,
                "n_quotes": int(counts[e.expiry]),
                "atm_vol": e.atm_vol,
                "skew_market": e.skew_market,
                "skew_rough": e.skew_rough,
                "skew_heston": heston_skew(heston.params, e.forward, e.T, e.r, e.atm_vol)
                if heston
                else np.nan,
                "rough_rmse": rough_rmse[e.expiry],
                "rough_shape": rough_shape[e.expiry],
                "heston_rmse": heston_rmse[e.expiry],
                "heston_shape": heston_shape[e.expiry],
            }
        )
    quote_rows = quotes.assign(trade_date=day, underlying=underlying).rename(columns={"model_iv": "rough_iv"})
    columns = [
        "trade_date",
        "underlying",
        "expiry",
        "T",
        "strike",
        "option_type",
        "iv",
        "rough_iv",
        "heston_iv",
    ]
    return row, expiries, quote_rows[columns].to_dict("records")


def _chains(root: Path, underlying: str):
    months: dict[str, pd.DataFrame | None] = {}

    def chain_for(day: date) -> pd.DataFrame:
        month = f"{day:%Y-%m}"
        if month not in months:
            path = store.chain_path(root, underlying, month)
            months[month] = (
                store.as_dates(pd.read_parquet(path), "trade_date", "expiry") if path.exists() else None
            )
        chains = months[month]
        return chains[chains["trade_date"] == day] if chains is not None else pd.DataFrame()

    return chain_for


def heston_run(job) -> tuple[dict, list[str]]:
    """Heston and SABR for a run of consecutive days of one underlying. Runs in a worker process."""
    root, underlying, days, previous_row = job
    previous = None
    if previous_row is not None and previous_row.get("fitted", False):
        try:
            previous = HestonParams(*(previous_row[k] for k in ("v0", "kappa", "theta", "xi", "rho")))
        except (ValueError, KeyError):
            previous = None
    chain_for = _chains(root, underlying)
    heston, sabr, failures = [], [], []
    for day in days:
        chain = chain_for(day)
        try:
            fit = fit_heston(chain, previous) if len(chain) else None
            sabr += sabr_rows(day, underlying, fit_sabr_chain(chain) if len(chain) else [])
        except Exception as error:  # report and keep going
            failures.append(f"{underlying} {day}: {type(error).__name__}: {error}")
            continue
        heston.append(heston_row(day, underlying, fit))
        if fit is not None:
            previous = fit.params
    return {store.HESTON_FITS: heston, store.SABR_FITS: sabr}, failures


def rough_run(job) -> tuple[dict, list[str]]:
    """Rough Bergomi (and same-window Heston) for a run of consecutive days. Runs in a worker."""
    root, underlying, days, previous_row = job
    previous = None
    if previous_row is not None and previous_row.get("fitted", False):
        try:
            previous = RoughBergomiParams(previous_row["H"], previous_row["eta"], previous_row["rho"])
        except (ValueError, KeyError):
            previous = None
    chain_for = _chains(root, underlying)
    fits, expiries, quotes, failures = [], [], [], []
    for day in days:
        chain = chain_for(day)
        try:
            fit = fit_rough_bergomi(chain, previous, n_paths=calibration.ROUGH_PATHS) if len(chain) else None
            heston = (
                fit_heston(chain, min_days=calibration.ROUGH_MIN_DAYS, max_days=calibration.ROUGH_MAX_DAYS)
                if fit
                else None
            )
            row, expiry_rows, quote_rows = rough_rows(day, underlying, fit, heston)
        except Exception as error:
            failures.append(f"{underlying} {day} (rough): {type(error).__name__}: {error}")
            continue
        fits.append(row)
        expiries += expiry_rows
        quotes += quote_rows
        if fit is not None:
            previous = fit.params
    return {store.ROUGH_FITS: fits, store.ROUGH_EXPIRIES: expiries, store.ROUGH_QUOTES: quotes}, failures


MODELS = {
    # name: (table that records which days are done, worker, underlyings, days per run)
    "heston": (store.HESTON_FITS, heston_run, None, 40),
    "rough": (store.ROUGH_FITS, rough_run, "ROUGH_UNDERLYINGS", 15),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--since", type=date.fromisoformat)
    parser.add_argument("--force", action="store_true", help="refit days that already have a fit")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--models", default=",".join(MODELS), help="comma-separated: " + ", ".join(MODELS))
    args = parser.parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if unknown := set(models) - set(MODELS):
        parser.error(f"unknown model(s): {', '.join(sorted(unknown))}")

    summary = store.read_summary(args.data_dir)
    if "n_quotes" in summary:
        summary = summary[summary["n_quotes"] > 0]
    if args.since:
        summary = summary[summary["trade_date"] >= args.since]

    failures = []
    for model in models:
        done_table, worker, underlyings_name, run_length = MODELS[model]
        allowed = getattr(config, underlyings_name) if underlyings_name else None
        existing = store.read_table(args.data_dir, done_table)
        done = set() if args.force else set(zip(existing["trade_date"], existing["underlying"], strict=True))
        jobs = []
        for underlying, rows in summary.groupby("underlying"):
            if allowed is not None and underlying not in allowed:
                continue
            pending = sorted(d for d in rows["trade_date"] if (d, underlying) not in done)
            fitted_before = existing[existing["underlying"] == underlying].sort_values("trade_date")
            for i in range(0, len(pending), run_length):
                days = pending[i : i + run_length]
                earlier = fitted_before[fitted_before["trade_date"] < days[0]]
                jobs.append(
                    (args.data_dir, underlying, days, earlier.iloc[-1].to_dict() if len(earlier) else None)
                )
        total = sum(len(j[2]) for j in jobs)
        if not total:
            print(f"{model}: nothing to fit.")
            continue

        print(
            f"{model}: fitting {total} underlying-days in {len(jobs)} run(s) with {args.workers} worker(s)..."
        )
        started, fitted = time.perf_counter(), 0
        # ProcessPoolExecutor rather than multiprocessing.Pool: if a worker process dies, it raises
        # BrokenProcessPool instead of quietly replacing the worker and waiting forever.
        pool = ProcessPoolExecutor(args.workers) if args.workers > 1 else None
        results = (
            (f.result() for f in as_completed([pool.submit(worker, j) for j in jobs]))
            if pool
            else map(worker, jobs)
        )
        for tables, failed in results:
            rows = tables[done_table]
            days = {(r["trade_date"], r["underlying"]) for r in rows}
            for relative, table_rows in tables.items():
                store.replace_days(args.data_dir, relative, pd.DataFrame(table_rows), days)
            fitted += len(rows)
            failures += failed
            rmse = [r["rmse"] for r in rows if r.get("fitted")]
            note = f"median RMSE {100 * np.median(rmse):.2f} pts" if rmse else "no fits"
            print(f"  {fitted}/{total} done ({time.perf_counter() - started:.0f}s); last run: {note}")
        if pool:
            pool.shutdown()
        print(f"{model}: fitted {fitted} underlying-day(s).")

    for failure in failures:
        print("FAILED:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
