"""The on-disk dataset: what the daily job writes and the app reads.

    <root>/summary.parquet                         one row per underlying per trading day
    <root>/chains/<UNDERLYING>/<YYYY-MM>.parquet   that month's implied-vol chains
    <root>/models/heston.parquet                   one Heston fit per underlying per day
    <root>/models/sabr.parquet                     one SABR fit per underlying, day and expiry

Writes are idempotent: saving a day replaces any rows already stored for it, so a re-run or a
backfill over existing data never duplicates anything. In production <root> is a checkout of the
orphan `market-data` branch.
"""

from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY = "summary.parquet"
HESTON_FITS = "models/heston.parquet"
SABR_FITS = "models/sabr.parquet"
REALIZED_WINDOW = 20  # trading days
# No index or large-cap stock moves this much in a day (|log return|), so a move this big that
# comes with a matching lot-size change is a bonus issue or split, not a price move.
CORPORATE_ACTION_MOVE = 0.15


def chain_path(root: Path, underlying: str, month: str) -> Path:
    return Path(root) / "chains" / underlying / f"{month}.parquet"


def as_dates(df: pd.DataFrame, *columns) -> pd.DataFrame:
    for col in columns:
        if col in df:
            df[col] = pd.to_datetime(df[col]).dt.date
    return df


def read_summary(root: Path) -> pd.DataFrame:
    path = Path(root) / SUMMARY
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "underlying"])
    return as_dates(pd.read_parquet(path), "trade_date")


def ingested(root: Path) -> set[tuple[date, str]]:
    """The (trade date, underlying) pairs already in the dataset."""
    summary = read_summary(root)
    return set(zip(summary["trade_date"], summary["underlying"], strict=True))


def read_chain(root: Path, underlying: str, day: date) -> pd.DataFrame:
    path = chain_path(root, underlying, f"{day:%Y-%m}")
    if not path.exists():
        return pd.DataFrame()
    chain = as_dates(pd.read_parquet(path), "trade_date", "expiry")
    return chain[chain["trade_date"] == day].reset_index(drop=True)


def _write(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd", index=False)


def save(root: Path, chains: list[pd.DataFrame], summary_rows: list[dict]):
    """Store some days' results: each (underlying, day) in `summary_rows` or `chains` replaces what's
    stored for it, chains included; an (underlying, day) with a summary row but no chain rows ends
    up with no quotes. To change only summary numbers, use write_summary."""
    root = Path(root)
    chains = [c for c in chains if len(c)]
    new = pd.concat(chains, ignore_index=True) if chains else None

    # Each (underlying, day) being saved replaces whatever is stored for it, even when its new
    # chain is empty (say, a re-run after tightening a filter), so stale quotes never linger.
    days = {(row["underlying"], row["trade_date"]) for row in summary_rows}
    if new is not None:
        days |= set(zip(new["underlying"], new["trade_date"], strict=True))
    files = defaultdict(set)
    for underlying, day in days:
        files[(underlying, f"{day:%Y-%m}")].add(day)

    for (underlying, month), dates in files.items():
        path = chain_path(root, underlying, month)
        parts = []
        if path.exists():
            old = as_dates(pd.read_parquet(path), "trade_date", "expiry")
            parts.append(old[~old["trade_date"].isin(dates)])
        if new is not None:
            parts.append(new[(new["underlying"] == underlying) & new["trade_date"].isin(dates)])
        parts = [p for p in parts if len(p)]
        if parts:
            _write(pd.concat(parts, ignore_index=True).sort_values(["trade_date", "expiry", "strike"]), path)
        elif path.exists():
            path.unlink()

    if summary_rows:
        new = pd.DataFrame(summary_rows)
        old = read_summary(root)
        if len(old):
            key = ["trade_date", "underlying"]
            replaced = old.set_index(key).index.isin(new.set_index(key).index)
            new = pd.concat([old[~replaced], new], ignore_index=True)
        write_summary(root, new)


def write_summary(root: Path, summary: pd.DataFrame):
    """Replace summary.parquet (recomputing realized vol); chains are left alone."""
    summary = with_realized_vol(summary.sort_values(["underlying", "trade_date"]).reset_index(drop=True))
    _write(summary, Path(root) / SUMMARY)


def adjusted_log_returns(spot: pd.Series, lot_size: pd.Series | None = None) -> pd.Series:
    """Daily log returns with bonus issues and splits taken out.

    A 1:1 bonus halves the share price and NSE doubles the lot size, so one contract is worth about
    the same. On such a day the return is replaced by the change in contract value. Lot-size
    revisions without a price jump (NSE resizes lots every so often) are left alone.
    """
    returns = np.log(spot).diff()
    if lot_size is None:
        return returns
    lot_change = np.log(lot_size.astype(float)).diff()
    action = (returns.abs() > CORPORATE_ACTION_MOVE) & ((returns + lot_change).abs() < 0.05)
    return returns.where(~action, returns + lot_change)


def with_realized_vol(summary: pd.DataFrame) -> pd.DataFrame:
    """Annualised close-to-close vol of the spot over the trailing window, per underlying."""
    summary = summary.drop(columns="rv_20d", errors="ignore")
    has_lots = "lot_size" in summary

    def realized(g: pd.DataFrame) -> pd.Series:
        lots = g["lot_size"].ffill().bfill() if has_lots and g["lot_size"].notna().any() else None
        returns = adjusted_log_returns(g["spot"], lots)
        return returns.rolling(REALIZED_WINDOW, min_periods=REALIZED_WINDOW).std() * np.sqrt(252)

    summary["rv_20d"] = pd.concat([realized(g) for _, g in summary.groupby("underlying", sort=False)])
    return summary


def read_table(root: Path, relative: str) -> pd.DataFrame:
    """A model-fit table (HESTON_FITS, SABR_FITS); empty if it doesn't exist yet."""
    path = Path(root) / relative
    if not path.exists():
        return pd.DataFrame(columns=["trade_date", "underlying"])
    return as_dates(pd.read_parquet(path), "trade_date", "expiry")


def replace_days(root: Path, relative: str, rows: pd.DataFrame, days: set[tuple[date, str]]):
    """Replace every row stored for the given (trade_date, underlying) pairs with `rows`."""
    old = read_table(root, relative)
    if len(old):
        keys = pd.MultiIndex.from_arrays([old["trade_date"], old["underlying"]])
        old = old[~keys.isin(list(days))]
    parts = [p for p in (old, rows) if len(p)]
    if not parts:
        return
    table = pd.concat(parts, ignore_index=True)
    order = [c for c in ("underlying", "trade_date", "expiry") if c in table]
    _write(table.sort_values(order).reset_index(drop=True), Path(root) / relative)
