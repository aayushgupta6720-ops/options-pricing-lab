# market-data

The dataset behind [options-pricing-lab](https://github.com/aayushgupta6720-ops/options-pricing-lab),
rebuilt every weekday evening from NSE's end-of-day F&O file by `.github/workflows/ingest.yml` on
`main`, which also fits the models. Every file is written by the scripts on `main`: the daily job
commits new days, and after a change to how quotes are cleaned or models are fitted, the scripts
are run over the whole history (`reclean_chains`, `rebuild_summary`, `calibrate --force`) and that
commit names the change. No number is edited by hand.

- `summary.parquet`: one row per underlying per trading day: spot, ATM implied vol at 7/30/60/90
  days, 25-delta skew at 30 days, the 30-day variance-swap vol replicated from the option strip
  (`vs_vol_30d`), 20-day realized vol, India VIX, quote counts.
- `chains/<UNDERLYING>/<YYYY-MM>.parquet`: every cleaned out-of-the-money quote (20+ trades, ₹0.50+,
  no stale close that leaves a static arbitrage) with its forward, log-moneyness, implied vol and delta.
- `models/heston.parquet`: one Heston fit per underlying per day (v0, kappa, theta, xi, rho, Feller
  ratio, fit error in vol, quote and expiry counts).
- `models/sabr.parquet`: one SABR fit per underlying, day and expiry (alpha, rho, nu with beta = 1,
  forward, fit error).
- `models/rough_bergomi.parquet`: one rough Bergomi fit per day for NIFTY (H, eta, rho, the fitted
  forward variance curve, fit and shape errors), the median-H fit of three with different random
  numbers (`H_seeds` holds all three), with Heston (one theta, and theta(t)) refitted to the same
  quotes for comparison. Times are in trading years (sessions / 252: `clock` = "trading").
- `models/rough_bergomi_expiries.parquet`: per day and expiry, the ATM skew of the market (from SABR),
  rough Bergomi and Heston, and each model's errors; `T` in trading years.
- `models/rough_bergomi_quotes.parquet`: every fitted quote with both models' implied vols.

Source: NSE's public archive (nsearchives.nseindia.com). For research and education.
