# How the numbers are made

Short notes on each method, with pointers to the code. Notation: spot $S$, strike $K$, years to expiry
$T$, continuously compounded rate $r$ and dividend yield $q$, volatility $\sigma$, forward
$F = S e^{(r-q)T}$, discount factor $D = e^{-rT}$.

## Black-Scholes-Merton (`optlab/models/black_scholes.py`)

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2), \qquad P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T$$

Greeks are the closed-form derivatives (delta $e^{-qT}N(d_1)$, gamma $e^{-qT}n(d_1)/(S\sigma\sqrt T)$,
vega $S e^{-qT} n(d_1)\sqrt T$, and so on). When $\sigma\sqrt T = 0$ the option is a forward that's
in or out of the money for sure, so $d_{1,2} = \pm\infty$; the code feeds those limits through the
same formulas rather than special-casing them.

**Checked against:** Hull's worked examples (call 4.76 / put 0.81; delta 0.522, gamma 0.066, vega
12.1, theta −4.31, rho 8.91), put-call parity to 1e-10, and finite-difference Greeks.

## Binomial tree (`optlab/models/binomial.py`)

Cox-Ross-Rubinstein: each step of length $\Delta t = T/n$ moves the spot up by $u = e^{\sigma\sqrt{\Delta t}}$
or down by $d = 1/u$, with risk-neutral probability $p = (e^{(r-q)\Delta t} - d)/(u - d)$. Values are
rolled back from the payoff at expiry; for American exercise each node takes the larger of its
continuation value and immediate exercise. The error versus Black-Scholes falls roughly like $1/n$ and
zig-zags depending on where the strike sits between nodes (see the *Model convergence* page).

**Checked against:** Hull's five-step American put (4.49); American call with no dividends equals the
European one; convergence to Black-Scholes.

## Monte Carlo (`optlab/models/monte_carlo.py`)

Under the risk-neutral measure, $S_T = S\exp\big((r - q - \sigma^2/2)T + \sigma\sqrt T Z\big)$ with
$Z \sim N(0,1)$, so a European price is $D\,\mathbb{E}[\text{payoff}(S_T)]$, estimated by a sample
mean with standard error $s/\sqrt{n}$. The engine is split into a path generator and a payoff pricer
so path-dependent options can reuse it.

**Checked against:** lands within 3 standard errors of Black-Scholes; the error shrinks like
$1/\sqrt n$; discounted paths are a martingale.

## Implied volatility (`optlab/implied_vol.py`)

The $\sigma$ that makes Black-Scholes match a price. Solved for a whole chain at once with a
bracketed Newton method: each quote keeps an interval that contains the root, takes a Newton step
$\sigma - (\text{BS}(\sigma) - P)/\text{vega}$ when it stays inside, and bisects otherwise. Prices
outside the no-arbitrage range return NaN. 200,000 quotes take about half a second, with errors
below 1e-10 for any quote worth at least the ₹0.05 tick.

## From NSE closing prices to a surface (`optlab/market/`)

1. **Data.** NSE's end-of-day F&O file ("bhavcopy"): one row per contract with the close (the
   volume-weighted average of the last half hour), open interest, volume and number of trades.
2. **Forward per expiry.** European put-call parity gives $C - P = D(F - K)$, so each strike where
   both legs trade gives $F = K + (C - P)/D$. The median over the strikes nearest the money is used.
   On 1 Oct 2026 this matched the NIFTY, BANKNIFTY and RELIANCE futures within 1–11 basis points.
   Solving for $D$ too (regressing $C - P$ on $K$) is far too noisy over a narrow strike range: it
   gave rates from −12% to +26%. So $r$ is fixed at the 91-day T-bill yield, and because forwards
   come from parity it only enters through discounting.
3. **Clean.** Keep out-of-the-money quotes (puts below the forward, calls above): they're the liquid
   side and their price is all time value. Require 20+ trades, a price of at least ₹0.50 (ten ticks),
   2+ days to expiry, and $|\ln(K/F)| \le 0.4$.
4. **Invert Black-76** ($C = D[F N(d_1) - K N(d_2)]$), which is Black-Scholes with $S = F$, $q = r$.
   NSE options are European, so this is exact rather than an approximation.
5. **Summarise.** ATM vol per expiry by interpolating at $\ln(K/F) = 0$; 25-delta vols by
   interpolating in forward delta. Fixed tenors interpolate ATM *total variance* $\sigma^2 T$
   linearly in $T$ (the usual way to avoid calendar arbitrage), with flat extrapolation only within
   14 days of a listed expiry. Realized vol is the annualised standard deviation of the last 20
   daily log returns, with bonus issues and splits taken out: a move over 15% that comes with a
   matching lot-size change (RELIANCE's 1:1 bonus halved the price and doubled the lot) counts only
   as the change in one contract's value.

**Sanity check:** NIFTY's 30-day ATM vol tracks India VIX, sitting a little below it, as expected
because VIX also prices the put wing.

## Strategy values (`optlab/strategy.py`)

Each leg is priced with Black-76 at its own strike's implied vol, read off that expiry's smile.
Before expiry, a spot move shifts the forward proportionally and each strike keeps its vol ("sticky
strike"). Max profit and loss come from evaluating the piecewise-linear payoff at each strike and at
spot 0, so a short put's worst case (spot to zero) isn't missed. Only a net long or short call
position is unlimited.
