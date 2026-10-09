"""Download NSE end-of-day option data and add it to the dataset.

    python -m scripts.ingest --data-dir data                     # catch up the last 10 days
    python -m scripts.ingest --data-dir data --date 2026-10-01   # one day
    python -m scripts.ingest --data-dir data --since 2024-07-01 --cache-dir .cache/bhavcopy  # backfill

Days already in the dataset are skipped (pass --force to redo them); days whose India VIX didn't
load are retried. Every calendar date is tried, since NSE occasionally trades on a weekend; a date
with no file (weekend, holiday, not published yet) is skipped, so the daily run can look back over
the last few days and fill in anything an earlier run missed. Any other failure is reported and
the run carries on with the remaining days, then exits with status 1.

A date with no file is remembered as closed once a later date has one (NSE publishes in order), and
isn't asked about again, so a run after the day is in makes no requests at all: NSE timing out at
night can't fail it. Today is only tried from 18:00 IST, when its files can exist. An explicit
--date, or --force, asks NSE regardless.
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
IST = ZoneInfo("Asia/Kolkata")
PUBLISHED_HOUR = 18  # IST; NSE posts a day's files in the evening, so before this today has none


def now() -> datetime:
    return datetime.now(IST)


def process_day(day: date, raw, vix: float) -> tuple[list, list[dict]]:
    """Chains and summary rows for every configured underlying.

    An underlying with no rows in the file (not listed yet, or suspended) still gets a summary row,
    with no spot, so the day counts as done for it.
    """
    chains, rows = [], []
    for underlying in config.UNDERLYINGS:
        rows_for = raw[raw["underlying"] == underlying]
        spot = rows_for["underlying_price"].dropna()
        lots = rows_for.loc[rows_for["instrument"] == "option", "lot_size"]
        chain = build_chain(raw, underlying)
        chains.append(chain)
        rows.append(
            {
                "trade_date": day,
                "underlying": underlying,
                "spot": float(spot.iloc[0]) if len(spot) else float("nan"),
                "lot_size": float(lots.mode().iloc[0]) if len(lots) else float("nan"),
                "india_vix": vix,
                "r": config.RISK_FREE_RATE,
                **daily_summary(chain),
            }
        )
    return chains, rows


def calendar_days(start: date, end: date):
    """Every date, weekends included: NSE sometimes trades on one (Budget day, Muhurat trading),
    and a date without a file is just skipped."""
    day = start
    while day <= end:
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
    parser.add_argument("--cache-dir", type=Path, help="keep the raw files here (useful for backfills)")
    parser.add_argument("--force", action="store_true", help="redo days already in the dataset")
    args = parser.parse_args(argv)

    current = now()
    today = current.date()
    if args.date:
        start = end = args.date
    else:
        last = today if current.hour >= PUBLISHED_HOUR else today - timedelta(days=1)
        start, end = args.since or today - timedelta(days=args.days), last
    closed = set() if args.force or args.date else store.closed_days(args.data_dir)

    done, retry_vix = set(), set()
    if not args.force:
        done = store.ingested(args.data_dir)
        summary = store.read_summary(args.data_dir)
        if "india_vix" in summary:
            retry_vix = set(summary.loc[summary["india_vix"].isna(), "trade_date"])

    session = requests.Session()
    chains, rows, month = [], [], None
    added, no_file, failed = 0, [], []

    for day in calendar_days(start, end):
        if day in closed:
            continue
        if all((day, u) in done for u in config.UNDERLYINGS) and day not in retry_vix:
            continue
        if month and f"{day:%Y-%m}" != month:  # flush each month so a long backfill can resume
            store.save(args.data_dir, chains, rows)
            chains, rows = [], []
        month = f"{day:%Y-%m}"

        requests_made = not nse.is_cached(day, args.cache_dir)
        try:
            raw = nse.load(day, config.UNDERLYINGS, args.cache_dir, session)
            try:
                vix = nse.india_vix(day, session, args.cache_dir)
            except (requests.RequestException, nse.BadResponse) as error:
                print(f"{day}: India VIX unavailable ({error}); will retry on the next run")
                vix = float("nan")
            day_chains, day_rows = process_day(day, raw, vix)
        except nse.NotPublished:
            no_file.append(day)
            if day.weekday() < 5:
                print(f"{day}: no file (holiday or not published yet)")
        except Exception as error:  # one bad day shouldn't cost the rest of the run
            print(f"{day}: FAILED: {type(error).__name__}: {error}")
            failed.append(day)
        else:
            chains += day_chains
            rows += day_rows
            added += 1
            parts = [
                f"{r['underlying']} {r['n_quotes']} quotes, 30d ATM {r['atm_iv_30d']:.1%}"
                for r in day_rows
                if r["n_quotes"]
            ]
            print(f"{day}: " + "; ".join(parts) + f"; India VIX {vix:.2%}")
        if requests_made:
            time.sleep(PAUSE_SECONDS)

    store.save(args.data_dir, chains, rows)
    latest = max((day for day, _ in store.ingested(args.data_dir)), default=None)
    now_closed = [day for day in no_file if latest and day < latest]
    store.add_closed_days(args.data_dir, now_closed)
    print(f"Added {added} day(s); {len(no_file)} date(s) had no file ({len(now_closed)} recorded as closed).")
    if failed:
        print(f"{len(failed)} day(s) failed: {', '.join(map(str, failed))}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
