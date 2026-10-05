"""The on-disk dataset: what the daily job writes and the app reads.

    <root>/summary.parquet                         one row per underlying per trading day
    <root>/chains/<UNDERLYING>/<YYYY-MM>.parquet   that month's implied-vol chains

Writes are idempotent: saving a day replaces any rows already stored for it, so a re-run or a
backfill over existing data never duplicates anything. In production <root> is a checkout of the
orphan `market-data` branch.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY = "summary.parquet"
REALIZED_WINDOW = 20  # trading days


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


def ingested_days(root: Path) -> set[date]:
    return set(read_summary(root)["trade_date"])


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
    root = Path(root)
    chains = [c for c in chains if len(c)]
    if chains:
        new = pd.concat(chains, ignore_index=True)
        new["month"] = pd.to_datetime(new["trade_date"]).dt.strftime("%Y-%m")
        for (underlying, month), rows in new.groupby(["underlying", "month"]):
            rows = rows.drop(columns="month")
            path = chain_path(root, underlying, month)
            if path.exists():
                old = as_dates(pd.read_parquet(path), "trade_date", "expiry")
                rows = pd.concat([old[~old["trade_date"].isin(set(rows["trade_date"]))], rows])
            _write(rows.sort_values(["trade_date", "expiry", "strike"]), path)

    if summary_rows:
        new = pd.DataFrame(summary_rows)
        old = read_summary(root)
        if len(old):
            key = ["trade_date", "underlying"]
            replaced = old.set_index(key).index.isin(new.set_index(key).index)
            new = pd.concat([old[~replaced], new], ignore_index=True)
        new = with_realized_vol(new.sort_values(["underlying", "trade_date"]).reset_index(drop=True))
        _write(new, root / SUMMARY)


def with_realized_vol(summary: pd.DataFrame) -> pd.DataFrame:
    """Annualised close-to-close vol of the spot over the trailing window, per underlying."""
    summary = summary.drop(columns="rv_20d", errors="ignore")
    log_returns = summary.groupby("underlying")["spot"].transform(lambda s: np.log(s).diff())
    rv = log_returns.groupby(summary["underlying"]).transform(
        lambda r: r.rolling(REALIZED_WINDOW, min_periods=REALIZED_WINDOW).std()
    )
    summary["rv_20d"] = rv * np.sqrt(252)
    return summary
