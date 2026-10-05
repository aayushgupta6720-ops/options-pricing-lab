"""Download NSE end-of-day option data and add it to the dataset.

    python -m scripts.ingest --data-dir data                     # catch up the last 10 days
    python -m scripts.ingest --data-dir data --date 2026-10-01   # one day
    python -m scripts.ingest --data-dir data --since 2024-07-01 --cache-dir .cache/bhavcopy  # backfill

Days already in the dataset are skipped (pass --force to redo them). A date with no file
(weekend, holiday, not published yet) is skipped, so the daily run can look back over the last
few days and fill in anything an earlier run missed. Any other HTTP failure stops the run with
an error, since it usually means NSE is blocking us.
"""

import argparse
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from optlab import config
from optlab.market import nse, store
from optlab.market.chain import build_chain
from optlab.surface import daily_summary

PAUSE_SECONDS = 1.0  # between uncached downloads, to stay polite to NSE


def process_day(day: date, raw, vix: float) -> tuple[list, list[dict]]:
    chains, rows = [], []
    for underlying in config.UNDERLYINGS:
        spot = raw.loc[raw["underlying"] == underlying, "underlying_price"]
        if spot.empty:
            continue
        chain = build_chain(raw, underlying)
        chains.append(chain)
        rows.append(
            {
                "trade_date": day,
                "underlying": underlying,
                "spot": float(spot.iloc[0]),
                "india_vix": vix,
                "r": config.RISK_FREE_RATE,
                **daily_summary(chain),
            }
        )
    return chains, rows


def trading_days(start: date, end: date):
    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--since", type=date.fromisoformat)
    parser.add_argument("--days", type=int, default=10, help="look-back when no date is given")
    parser.add_argument("--cache-dir", type=Path, help="keep the raw zips here (useful for backfills)")
    parser.add_argument("--force", action="store_true", help="redo days already in the dataset")
    args = parser.parse_args(argv)

    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    if args.date:
        start = end = args.date
    else:
        start, end = args.since or today - timedelta(days=args.days), today

    done = set() if args.force else store.ingested_days(args.data_dir)
    session = requests.Session()
    chains, rows, month = [], [], None
    added = skipped = 0

    for day in trading_days(start, end):
        if day in done:
            continue
        if month and f"{day:%Y-%m}" != month:  # flush each month so a long backfill can resume
            store.save(args.data_dir, chains, rows)
            chains, rows = [], []
        month = f"{day:%Y-%m}"

        cached = args.cache_dir and (args.cache_dir / f"fo_{day:%Y%m%d}.csv.zip").exists()
        try:
            raw = nse.load(day, config.UNDERLYINGS, args.cache_dir, session)
        except nse.NotPublished:
            print(f"{day}: no file (holiday or not published yet)")
            skipped += 1
            continue
        vix = nse.india_vix(day, session)
        day_chains, day_rows = process_day(day, raw, vix)
        chains += day_chains
        rows += day_rows
        added += 1
        parts = [f"{r['underlying']} {r['n_quotes']} quotes, 30d ATM {r['atm_iv_30d']:.1%}" for r in day_rows]
        print(f"{day}: " + "; ".join(parts) + f"; India VIX {vix:.2%}")
        if not cached:
            time.sleep(PAUSE_SECONDS)

    store.save(args.data_dir, chains, rows)
    print(f"Added {added} day(s); {skipped} date(s) had no file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
