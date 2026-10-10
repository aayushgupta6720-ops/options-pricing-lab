# market-data

The dataset behind [options-pricing-lab](https://github.com/aayushgupta6720-ops/options-pricing-lab),
rebuilt every weekday evening from NSE's end-of-day F&O file by `.github/workflows/ingest.yml` on
`main`, which also fits the models. Nothing here is edited by hand.

- `summary.parquet`: one row per underlying per trading day: spot, ATM implied vol at 7/30/60/90
  days, 25-delta skew at 30 days, the 30-day variance-swap vol replicated from the option strip
  (`vs_vol_30d`), 20-day realized vol, India VIX, quote counts.
- `chains/<UNDERLYING>/<YYYY-MM>.parquet`: every cleaned out-of-the-money quote with its forward,
  log-moneyness, implied vol and delta.
- `models/heston.parquet`: one Heston fit per underlying per day (v0, kappa, theta, xi, rho, Feller
  ratio, fit error in vol, quote and expiry counts).
- `models/sabr.parquet`: one SABR fit per underlying, day and expiry (alpha, rho, nu with beta = 1,
  forward, fit error).
- `models/rough_bergomi.parquet`: one rough Bergomi fit per day for NIFTY (H, eta, rho, the fitted
  forward variance curve, fit and shape errors), with Heston refitted to the same quotes for comparison.
- `models/rough_bergomi_expiries.parquet`: per day and expiry, the ATM skew of the market (from SABR),
  rough Bergomi and Heston, and each model's errors.
- `models/rough_bergomi_quotes.parquet`: every fitted quote with both models' implied vols.

Source: NSE's public archive (nsearchives.nseindia.com). For research and education.
