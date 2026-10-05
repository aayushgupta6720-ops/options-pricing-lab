# Options Pricing & Volatility Lab

Option pricing models built and tested from scratch, applied to real NSE data: a daily implied
volatility surface for NIFTY, BANKNIFTY and RELIANCE, rebuilt from NSE's official end-of-day files
with a history back to July 2024.

**Live demo:** _coming soon (Render)_

![Volatility surface page](docs/screenshots/surface.png)

## What's in it

| Page | What it shows |
|---|---|
| **Volatility surface** | For any trading day: the smile per expiry, ATM and 25-delta term structure, and a 3D surface. Spot, 30-day ATM vol, 25Δ skew, realized vol and India VIX with day-on-day changes. |
| **Volatility history** | 30-day implied vs 20-day realized vol vs India VIX, the term structure and skew over time, and how often implied vol overpriced the volatility that followed. |
| **Strategy payoff** | Multi-leg positions (spreads, straddles, condors, butterflies, or your own legs) on the latest close. Each leg is priced at its strike's implied vol, with P&L in ₹ per lot, breakevens, max profit/loss and position Greeks. |
| **Pricer** | One option priced by Black-Scholes, a binomial tree and Monte Carlo side by side, with Greeks, the early-exercise premium for American options, and an implied-vol calculator. Loads the latest NIFTY at-the-money option in one click. |
| **Greeks** | Delta, gamma, vega, theta and rho against spot, volatility or time to expiry. |
| **Model convergence** | The tree's error against steps (≈1/n) and Monte Carlo's confidence interval against paths (≈1/√n), measured against the exact Black-Scholes price. |

## Findings from the data

From 560 trading days, 1 Jul 2024 to 5 Oct 2026, all computed from the published dataset:

- **The pipeline tracks India VIX.** NIFTY's 30-day ATM vol has a 0.975 correlation with India VIX.
  VIX sits above it on 95% of days, by about 1 point. That's expected: VIX averages over the whole
  strip of strikes, including the expensive put wing.
- **Implied vol usually overprices what follows.** 30-day ATM vol exceeded the next 20 days'
  realized vol on 68% of days for NIFTY (median gap +1.4 points), 66% for BANKNIFTY (+1.6) and 58%
  for RELIANCE (+0.7): the variance risk premium that option sellers collect.
- **Index puts carry more crash premium than single-stock puts.** NIFTY's 25-delta skew averaged
  2.2 vol points against 1.2 for RELIANCE.
- **Stress inverts the term structure.** On the 10% of days with the highest NIFTY 30-day vol, 90-day
  vol was below 30-day vol every time; across all days it sloped upward only half the time. 30-day vol
  ranged from 8.6% (24 Dec 2025) to 29.1% (30 Mar 2026).
- **Corporate actions need handling.** RELIANCE's 1:1 bonus on 28 Oct 2024 halved the price and
  doubled the lot size. Taken at face value, 20-day realized vol reads 245%; the pipeline spots a
  price jump matched by a lot-size change and counts only the change in contract value.

| Volatility history (dark theme) | Strategy payoff |
|---|---|
| ![History page](docs/screenshots/history-dark.png) | ![Strategy page](docs/screenshots/strategy.png) |

## How it works

```
NSE bhavcopy (daily zip) ──► optlab/market/nse.py      parse options + futures
                         ──► optlab/market/forwards.py forward per expiry from put-call parity
                         ──► optlab/market/chain.py    clean OTM quotes, invert Black-76 → IV
                         ──► optlab/surface.py         ATM / 25Δ / fixed-tenor summary
                         ──► market-data branch        parquet, one file per underlying-month
                                     │
               GitHub Actions, weekdays 20:17 IST      ▼
                                              Streamlit app (Render)
```

- **`optlab/`** is a plain Python library with no Streamlit in it: the pricing models
  (`models/`), implied vol, finite-difference Greeks, the NSE pipeline (`market/`) and strategy maths.
- **`scripts/ingest.py`** adds trading days to the dataset. It's idempotent, skips holidays, and
  looks back 10 days so a missed day gets filled in by the next run.
- **`.github/workflows/ingest.yml`** runs it every weekday evening and commits to the
  `market-data` branch. The app reads that branch, so it never calls NSE itself.
- The maths, and what each piece is checked against, is in [docs/models.md](docs/models.md).

## Testing

71 tests run in CI (`ruff` + `pytest`, about 2 seconds, no network):

- **Models:** Hull's textbook values for prices, Greeks and the 5-step American put; put-call
  parity; tree → Black-Scholes convergence; Monte Carlo within 3 standard errors; analytic vs
  finite-difference Greeks.
- **Implied vol:** round trips over a grid of strikes, maturities and vols; NaN outside
  no-arbitrage bounds.
- **Pipeline:** parsing a real (trimmed) bhavcopy; parity forwards within 10 bp of the futures;
  recovering a known smile from synthetic prices; idempotent storage; realized vol against its
  definition, including across a bonus issue and a lot-size revision.
- **Strategies:** breakevens, bounded vs unlimited P&L (including a short put's spot-to-zero case),
  and that an iron condor collects a credit.
- **App:** every page renders without an exception (Streamlit AppTest), and the pricer reproduces
  the NSE close at the market's implied vol.

## Quick start

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements-dev.txt
./venv/bin/python -m pytest -q
./venv/bin/streamlit run app.py   # reads the published dataset from GitHub
```

To work with a local copy of the dataset, check out the data branch into `data/` (the app prefers
it when present), or rebuild it from NSE:

```bash
git worktree add data market-data
# or, from scratch (~570 trading days, about an hour; keeps the raw zips in .cache/):
./venv/bin/python -m scripts.ingest --data-dir data --since 2024-07-01 --cache-dir .cache/bhavcopy
```

## Data caveats

- **End-of-day only.** The bhavcopy has closing prices, not bid/ask, so the surface updates daily.
  Closes are NSE's volume-weighted average of the last half hour, which keeps options and the
  underlying roughly in step.
- **Liquidity filter.** Quotes need 20+ trades and a price of at least ₹0.50, so far wings and
  long-dated expiries are thin. NIFTY lists options out to 2031 but only the first year or so trades.
- **Rate.** A single 5.5% rate (the 91-day T-bill in September 2026) is used throughout. Forwards come
  from put-call parity, so the rate only affects discounting, which moves a 30-day vol by well
  under 1% of its value even across 2024's higher rates.
- **Source.** NSE's public archive, for research and education. Not investment advice.

## Roadmap

- **Heston and SABR calibration** to each day's surface, with model-vs-market residuals and
  parameter stability over the backfilled history.
- **Exotic options** (Asian, barrier, lookback) on the Monte Carlo engine, with variance reduction
  (antithetic, control variates, Sobol) and numba speed-ups.
