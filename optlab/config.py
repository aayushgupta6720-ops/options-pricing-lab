"""Settings shared by the data pipeline and the app."""

UNDERLYINGS = ("NIFTY", "BANKNIFTY", "RELIANCE")

# Discount rate, continuously compounded. Forwards come from put-call parity, so this only enters
# through discounting: a 1-point error moves a 30-day implied vol by under 0.2% of its value.
# 91-day T-bill cut-off yield on 30 Sep 2026 was 5.52% (RBI auction). Rates were up to ~1.3 points
# higher in 2024, which is inside that tolerance for the backfilled history.
RISK_FREE_RATE = 0.055

# Which quotes count. Close prices are NSE's volume-weighted average of the last half hour.
MIN_DAYS_TO_EXPIRY = 2  # calendar days; expiry-week quotes are dominated by gamma and ticks
MIN_TRADES = 20  # trades in the day for a quote to count as a real price
MIN_PRICE = 0.5  # rupees; the tick is 0.05, so below this the tick alone is a 10% move
MAX_ABS_LOG_MONEYNESS = 0.4  # |ln(K / F)|
PARITY_STRIKES = 8  # strike pairs nearest the money used to infer each expiry's forward

# Earliest bhavcopy in the format nse.py reads (2023 dates return 404).
FIRST_AVAILABLE_DAY = "2024-07-01"

# Fixed tenors for the daily summary, in calendar days.
TENORS = (7, 30, 60, 90)
