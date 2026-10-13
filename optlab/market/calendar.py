"""Trading time: maturities counted in NSE sessions rather than calendar days.

Calendar time gives a weekend two days of variance it doesn't have: on Fridays NIFTY's 7-day ATM vol
averaged 0.950 of its 30-day vol, and on Mondays 1.025, because a Friday's week-out expiry spans a
weekend and Monday's doesn't. Counting sessions puts every day with a price on the same footing. The
rough Bergomi fits use it (calibration.to_trading_time), since their findings live at the short end.
"""

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

SESSIONS_PER_YEAR = 252


def _weekdays_after(start: date, end: date) -> int:
    """Weekdays in (start, end]."""
    if end <= start:
        return 0
    return int(np.busday_count(start + timedelta(days=1), end + timedelta(days=1)))


def sessions_between(start: date, end: date, sessions: Sequence[date]) -> int:
    """NSE sessions after `start` up to and including `end`. `sessions` are the days known to have
    traded (sorted): up to the last of them they're the count, holidays and all; past it, weekdays
    are, since holidays to come aren't in the data."""
    if end <= start:
        return 0
    known = np.asarray(sessions, dtype="datetime64[D]")
    if len(known) == 0:
        return _weekdays_after(start, end)
    last = known[-1].astype(date)
    upto = min(end, last)
    lo, hi = np.datetime64(start, "D"), np.datetime64(upto, "D")
    counted = int(np.searchsorted(known, hi, side="right") - np.searchsorted(known, lo, side="right"))
    return counted + _weekdays_after(max(start, last), end)
