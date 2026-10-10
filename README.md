# Options Pricing & Volatility Lab

Option pricing models built and tested from scratch, applied to real NSE data: a daily implied
volatility surface for NIFTY, BANKNIFTY and RELIANCE, rebuilt from NSE's official end-of-day files
with a history back to July 2024.

**Live demo:** https://options-pricing-lab.onrender.com. It runs on Render's free plan, which sleeps
after 15 minutes idle, so the first visit can take about a minute to wake up. The option data updates
every weekday evening; the Option pricer and Strategy builder use the live index level during market
hours.

![Implied volatility page](docs/screenshots/surface.png)

## What's in it

Five pages; the bigger ones are split into tabs, and only the open tab is computed. Each tab has its
own link, e.g. [/pricer?tab=Convergence](https://options-pricing-lab.onrender.com/pricer?tab=Convergence).

| Page | What it shows |
|---|---|
| **Implied volatility** | *Surface:* for any trading day, the smile per expiry, the at-the-money and 25-delta term structure, and a 3D surface, with spot, 30-day implied vol, 30-day skew, realized vol and India VIX and their day-on-day changes. *History:* 30-day at-the-money and variance-swap vol vs 20-day realized vol vs India VIX, the term structure and skew over time, and the variance premium against the volatility that followed. |
| **Models vs market** | *Heston & SABR:* Heston (one set of five parameters per day for the whole surface) and SABR (one smile per expiry) fitted to every trading day: the fitted parameters, each model against the market's smile, a heatmap of where Heston misses, SABR's parameters by expiry, and how Heston's parameters moved over two years. *Rough volatility:* rough Bergomi against Heston on NIFTY's short expiries: the at-the-money skew against maturity on log-log axes (a power law is a straight line), short-expiry smiles under both models, which model gets the smile's shape right by maturity, and the fitted roughness H over time. |
| **Strategy builder** | Multi-leg positions (spreads, straddles, condors, butterflies, or your own legs) at the live index level for NIFTY and BANKNIFTY, else the last close (RELIANCE always uses the close). Each leg is priced at its strike's implied vol, with P&L in ₹ per lot, breakevens, max profit/loss and position Greeks. |
| **Option pricer** | One option, set in the sidebar or loaded in one click as the at-the-money NIFTY option at the live index level. *Prices:* Black-Scholes, a binomial tree and Monte Carlo side by side, Greeks, the early-exercise premium for American options, and an implied-vol calculator. *Greeks:* each Greek against spot, volatility or time to expiry. *Convergence:* the tree's error against steps (≈1/n) and Monte Carlo's confidence interval against paths (≈1/√n). *Heston model:* sliders for Heston's five parameters (preloaded with the latest NIFTY fit), the smile and term structure they produce, and a Monte Carlo check against the formula. |
| **Exotic options** | Asian, barrier and lookback options on the sidebar's option, priced by Monte Carlo under Black-Scholes, Heston and rough Bergomi side by side, with the Black-Scholes closed form as a check, and five variance-reduction methods compared, with each one's error against the number of paths. |

Labels use plain names (mean reversion, vol of vol); the symbols and the method behind each chart
are one click away, in the help icons and each page's *How this is computed* section.

## Findings from the data

From 562 trading days, 1 Jul 2024 to 5 Oct 2026, all computed from the published dataset:

- **The pipeline tracks India VIX.** NIFTY's 30-day ATM vol has a 0.975 correlation with India VIX.
  VIX sits above it on 95% of days, by about 1 point. That's expected: VIX averages over the whole
  strip of strikes, including the expensive put wing.
- **The variance risk premium is there, but 27 months can't pin it down.** Selling 30-day variance
  at the variance-swap rate, replicated from each day's whole option strip as India VIX is, earned
  about one vol point against the variance that followed: +0.95 for NIFTY, +0.78 for BANKNIFTY and
  +1.34 for RELIANCE (546 days with a known outcome, to 9 Oct 2026). Consecutive days share most of
  their 20-day window, so that's only about 27 independent periods, and the 95% block-bootstrap
  intervals include zero for both indices (NIFTY −0.5 to +2.4). At the money there's no premium at
  all (−0.35 to +0.28): it sits in the put wing. The familiar statistic, implied vol above what
  followed on 67% of NIFTY days, says little: realized vol is right-skewed, and a constant forecast
  at its average beats it on 66%.
- **Index puts carry more crash premium than single-stock puts.** NIFTY's 25-delta skew averaged
  2.2 vol points against 1.2 for RELIANCE.
- **Stress inverts the term structure.** On the 10% of days with the highest NIFTY 30-day vol, 90-day
  vol was below 30-day vol every time; across all days it sloped upward only half the time. 30-day vol
  ranged from 8.6% (24 Dec 2025) to 29.1% (30 Mar 2026).
- **Corporate actions need handling.** RELIANCE's 1:1 bonus on 28 Oct 2024 halved the price and
  doubled the lot size. Taken at face value, 20-day realized vol reads 245%; the pipeline spots a
  price jump matched by a lot-size change and counts only the change in contract value.

### What the Heston and SABR fits show

Heston fitted to expiries from a week to a year, SABR to each expiry, both between the 5-delta put and
call, on every trading day since July 2024:

- **SABR fits a single smile almost exactly; Heston fits the whole surface well.** Median error is
  0.11 vol points for SABR (0.22 for RELIANCE) against 0.25–0.39 for Heston, which has to explain
  every expiry with one set of five numbers.
- **Heston's spot vol is the market's short-dated vol.** √v₀ has a 0.98 correlation with NIFTY's
  7-day ATM vol and 0.95 with India VIX.
- **Stress steepens the skew.** NIFTY's spot-vol correlation ρ has a median of −0.38, falling to
  −0.55 on the 10% of days with the highest 30-day vol; Heston's fit error also rises (0.33 → 0.57
  points). RELIANCE's ρ is only −0.12: single-stock smiles are flatter.
- **The Feller condition fails almost every day** (96–100% of days): matching an index skew needs a
  vol of vol large enough for the variance to touch zero.
- **Monthly-only underlyings can't pin down mean reversion.** κ sits at its cap of 30 (an 8-day
  half-life) on 57% of RELIANCE days and 18% of BANKNIFTY days, against 3% for NIFTY with its weekly
  expiries. Letting it go higher lowers the error a little at the price of parameters like a
  one-day half-life.
- **Expiries under a week need jumps.** Averaged over all NIFTY days, Heston's vols for expiries under
  a week (left out of the fit) are 1.2–1.3 points too low in the wings and 0.3 too high at the money:
  those smiles are steeper than any diffusion produces. Inside the fitted range the average miss in
  any region is under 0.45 points.
- **The far wings are lottery tickets.** Beyond the 5-delta wings, options trade at a few rupees at
  vols no diffusion reaches (a one-week NIFTY put 11 standard deviations out closed at ₹0.85, an
  implied vol of 52%). In a first run that fitted them too, they caused every one of the worst NIFTY
  fits and pushed κ to its cap on 74% of BANKNIFTY and 87% of RELIANCE days. Both fits now leave them out.

### Rough volatility on NIFTY

Rough Bergomi and Heston fitted each day to the same NIFTY expiries (2 to 91 days, between the 5-delta
wings), over all 562 days:

- **Rough Bergomi fits the smiles better, most of all the shortest.** Its *shape error* (each expiry's
  misses after removing their average, the fair comparison since rough Bergomi fits one variance level
  per expiry) is lower than Heston's on 89% of days: 0.28 against 0.39 vol points for expiries of a week
  or less (better on 84% of them), 0.17–0.18 against 0.20–0.22 out to three months.
- **But NIFTY's skew doesn't steepen as fast as rough volatility predicts.** Across 2–91 days the
  at-the-money skew falls with slope −0.29 in maturity (H ≈ 0.21 read through the power law), and inside
  two weeks only −0.21, about Heston's −0.20 and well short of rough Bergomi's −0.38. Rough Bergomi's
  edge comes from each short expiry's smile shape (curvature and wings), not from the term structure of
  skew.
- **The fitted roughness is low and noisy.** H has a median of 0.07 (middle half 0.03–0.19), below the
  0.21 the skew slope implies, and moves a lot day to day. It's higher on stressed days (0.21 in the
  most volatile tenth against 0.06 on calm ones), when both models fit worse (0.35 and 0.57 points).
- **The variance-swap replication reproduces India VIX.** The 30-day variance swap from our own strips
  has a 0.979 correlation with India VIX and sits 0.30 points below it on average (the strip stops at
  the quoted strikes).

### Exotics

- **Barriers depend on the smile, not just the ATM vol.** On 22-day NIFTY options (5 Oct 2026, the
  latest Heston and rough Bergomi fits, Black-Scholes at the 14.1% ATM vol), an up-and-out call with its
  barrier 4.6% above spot is worth ₹109 under Black-Scholes and ₹155 under both Heston and rough Bergomi;
  a down-and-in put 4.7% below spot ₹159 against ₹205 and ₹198; a floating lookback put ₹593 against
  ₹525 and ₹538. The two smile-aware models agree within Monte Carlo error; flat volatility is what's
  off. Asians move much less (3–4%).
- **Variance reduction depends on the option, and for Sobol on the path count.** For an arithmetic
  Asian under Black-Scholes, a geometric-Asian control variate cuts the variance per path by a factor
  that holds at any number of paths but swings with the option: about 1,300× for a 1-year
  at-the-money Asian, 50,000× for a 22-day at-the-money NIFTY one, 2,000× for one 5% out of the money.
  Sobol points with a principal-component path construction aren't a fixed factor at all: their error
  falls faster than one over the square root of the paths, so their gain grows with them (330× at
  4,000 paths, 9,600× at 32,000 for the 1-year Asian). The Exotic options page plots each method's
  error against paths. Only the Asian has a control this good; the barrier, lookback, Heston and
  rough Bergomi prices use plain Monte Carlo.

| Implied volatility, History tab (dark theme) | Strategy builder |
|---|---|
| ![History page](docs/screenshots/history-dark.png) | ![Strategy page](docs/screenshots/strategy.png) |

## How it works

```
NSE bhavcopy (daily zip) ──► optlab/market/nse.py      parse options + futures
                         ──► optlab/market/forwards.py forward per expiry from put-call parity
                         ──► optlab/market/chain.py    clean OTM quotes, invert Black-76 → IV
                         ──► optlab/surface.py         ATM / 25Δ / fixed-tenor summary
                         ──► market-data branch        parquet, one file per underlying-month
                         ──► optlab/calibration.py     Heston per day, SABR per expiry, rough
                                                       Bergomi per day (NIFTY) → models/
                                     │
               GitHub Actions, weekday evenings        ▼
                                              Streamlit app (Render)
```

- **`optlab/`** is a plain Python library with no Streamlit in it: the pricing models
  (`models/`), implied vol, finite-difference Greeks, the NSE pipeline (`market/`) and strategy maths.
- **`scripts/ingest.py`** adds trading days to the dataset. It's idempotent, skips holidays, and
  looks back 10 days so a missed day gets filled in by the next run. It tries every calendar date,
  because NSE occasionally trades on a weekend (Budget day 2025 was a Saturday). One failing day
  doesn't stop the others; the run reports it and exits non-zero. A date with no file is recorded in
  `closed_days.csv` once a later date has one, and isn't asked about again; today is only tried from
  18:00 IST. So once the day is in, later runs make no requests, and NSE's archive timing out late at
  night (as it did on 9 Oct 2026) can't fail them.
- **`scripts/calibrate.py`** fits Heston and SABR (every underlying) and rough Bergomi (NIFTY, by
  Monte Carlo) to every day that doesn't have a fit yet, warm-starting each from the previous day's.
  A new day takes about 15 seconds; the whole history about two minutes for Heston and SABR and under
  70 minutes for rough Bergomi on 8 cores.
- **`scripts/rebuild_summary.py`** recomputes the daily summary from the stored quotes after a
  change to how it's derived, without downloading anything.
- **`.github/workflows/ingest.yml`** runs ingest and calibration every weekday evening and commits to the
  `market-data` branch. The app reads its end-of-day data from that branch. It's scheduled at 20:17 IST
  and again at 22:17, 00:17, 02:17 and 08:17, because GitHub's scheduled runs can start hours late or
  not at all; a run that finds the day already done saves nothing. In practice they've started 3 to 7
  hours late, so a launchd job on my Mac also starts the workflow (`gh workflow run ingest.yml`) at
  20:17 and 22:17 IST on weekdays; GitHub's schedule covers the evenings the Mac is off.
- **`optlab/market/live.py`** is the one live call: NIFTY's and BANKNIFTY's level from NSE's website
  API, else Yahoo Finance. RELIANCE stays at its last close: NSE refuses scripts its stock quotes, and
  Yahoo turned the app's server away from its first request. The app asks
  at most once a minute per underlying, leaves a source alone for 10 minutes after it refuses, and
  uses the price only when it's later than the last close in the dataset; otherwise, and whenever
  neither source answers, it falls back to that close. A live option takes its vol from the last
  close's smile at the same moneyness, and the Option pricer and Strategy builder say so.
- The maths, and what each piece is checked against, is in [docs/models.md](docs/models.md).

## Testing

232 tests run in CI (`ruff` + `pytest`, about 90 seconds, no network):

- **Models:** Hull's textbook values for prices, Greeks and the 5-step American put; put-call
  parity; tree → Black-Scholes convergence, including the low-vol cases where the tree switches
  lattice; tree delta and gamma against Black-Scholes and a fine tree; Monte Carlo within 3
  standard errors; analytic vs finite-difference Greeks; American implied vol round trips.
- **Heston and SABR:** the published Fang–Oosterlee reference price (5.785155450) to 1e-7; the fast
  integration against adaptive quadrature from 2 days to 2 years; put-call parity; the Black-Scholes
  limit; Monte Carlo against the formula; SABR's flat, symmetric and ATM limits. Calibration recovers
  known Heston and SABR parameters from synthetic markets, ignores corrupted far-wing quotes, and fits
  a real NIFTY day to under 1.5 points.
- **Rough Bergomi:** E[S_T] = S₀ and E[V_t] = ξ₀(t); the Volterra process's variance is t^(2H); at
  H = ½ it's exactly a Brownian motion; the simulated ATM skew falls with slope near H − ½ for H = 0.1
  and stays flat for H = ½; common random numbers make prices smooth in the parameters. Calibration
  recovers H, ρ and the forward variance from a synthetic rough market simulated with different
  random numbers.
- **Variance swaps:** the replication formula recovers Black-Scholes variance to 0.5% and lands near
  India VIX on a real day.
- **Exotics:** Monte Carlo against the closed forms for geometric Asians, floating lookbacks and all
  eight barrier types; knock-in + knock-out = vanilla; the numba kernels against numpy; Heston and rough
  Bergomi paths reduce to Black-Scholes in their limits; every variance-reduction method agrees with
  plain Monte Carlo and the control variates beat it by orders of magnitude.
- **Implied vol:** round trips over a grid of strikes, maturities and vols; NaN outside
  no-arbitrage bounds.
- **Live prices:** NSE first and Yahoo as the fallback, timestamps in IST, a refusing source (403,
  429, an HTML page instead of JSON) resting, and the app ignoring live prices that are no newer than
  the close or implausibly far from it; stocks never looked up. The Strategy builder keeps the price
  it took until Refresh.
- **Pipeline:** parsing a real (trimmed) bhavcopy; NSE's HTTP handling (404 skip, 403 fail-fast,
  429/5xx retry, HTML-instead-of-zip); parity forwards within 10 bp of the futures; recovering a
  known smile from synthetic prices; the ATM gap gate and order-independent 25-delta vols;
  idempotent storage; realized vol across a bonus issue and a lot-size revision; ingest with
  weekend sessions, a failing day mid-run, a newly added underlying, VIX retries, month flushes,
  closed days remembered (and not closed until a later day has a file) and today waiting for 18:00 IST.
- **Strategies:** breakevens, bounded vs unlimited P&L (including a short put's spot-to-zero case),
  that an iron condor collects a credit, and theta (with the forward rolling down) against the
  value a moment later.
- **App:** every page and tab renders (Streamlit AppTest), and only the open tab runs. Plus
  regressions for state and edge cases: sidebar inputs surviving a trip to a market page, choices
  inside a tab surviving a visit to another tab, per-underlying strategy widths, low-vol and
  American inputs, a missing latest chain, an underlying with no data, and working offline. The
  tests point the data URL at a closed port, so they can't quietly fetch from GitHub.

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

- **End-of-day smiles, live index level.** The bhavcopy has closing prices, not bid/ask, so the
  surface, the smiles and the model fits update once a day. Closes are NSE's volume-weighted average
  of the last half hour, which keeps options and the underlying roughly in step. During market hours
  the Option pricer and Strategy builder move NIFTY and BANKNIFTY to the live index level, but still
  read volatility off the last close's smile.
- **Liquidity filter.** Quotes need 20+ trades and a price of at least ₹0.50, so far wings and
  long-dated expiries are thin. NIFTY lists options out to 2031 but only the first year or so trades.
- **Rate.** A single 5.5% rate (the 91-day T-bill in September 2026) is used throughout. Forwards come
  from put-call parity, so the rate only affects discounting, which moves a 30-day vol by well
  under 1% of its value even across 2024's higher rates.
- **Source.** NSE's public archive, for research and education. Not investment advice.

## Roadmap

- Rough Heston (via its fractional Riccati equation) as a second rough model with a semi-closed form.
- Fitting rough Bergomi to all three underlyings, not just NIFTY.
