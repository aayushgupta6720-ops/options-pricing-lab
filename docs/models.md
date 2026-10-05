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

CRR needs $d < e^{(r-q)\Delta t} < u$, which fails when the drift per step outruns the volatility
(low vol, high rates, long maturities, few steps). The tree then switches to a drift-centred lattice,
$u, d = e^{(r - q - \sigma^2/2)\Delta t \pm \sigma\sqrt{\Delta t}}$, whose probability is always in
$(0, 1)$.

**Greeks from the tree.** Delta and gamma are read off the first two steps' nodes (Hull's method).
Bumping spot by less than the node spacing and re-running doesn't work: at that scale the tree price
is piecewise linear in $S$, so a second difference comes out as zero or explodes. Vega, theta and rho
re-price the tree with bumped inputs.

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
5. **Summarise.** ATM vol per expiry by interpolating at $\ln(K/F) = 0$, but only when the quotes
   either side of the forward are within one standard deviation ($\sigma\sqrt T$) of each other;
   otherwise a thin expiry would interpolate straight across the skew. 25-delta vols: find the
   strike where forward delta hits ±0.25 (with delta forced to fall as the strike rises, since noisy
   quotes can break that order), then read the vol off the smile there. Fixed tenors interpolate ATM *total variance* $\sigma^2 T$
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
strike"). The carry rate $b = \ln(F/S)/T$ is held constant, so as time passes the forward rolls down
towards spot ($F = S e^{b\tau}$); theta includes that roll-down. Max profit and loss come from evaluating the piecewise-linear payoff at each strike and at
spot 0, so a short put's worst case (spot to zero) isn't missed. Only a net long or short call
position is unlimited.

## Heston (`optlab/models/heston.py`)

$$\frac{dS}{S} = (r - q)\,dt + \sqrt{v}\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \qquad d\langle W_1, W_2\rangle = \rho\,dt$$

Variance starts at $v_0$, is pulled towards $\theta$ at speed $\kappa$ (half-life $\ln 2/\kappa$
years) and fluctuates with vol of vol $\xi$. A negative $\rho$ makes vol rise when the market falls,
which tilts the smile towards downside strikes; $\xi$ curves it.

**Pricing.** The characteristic function of $X = \ln(S_T/F)$ is $\phi(u) = e^{C(u) + D(u)v_0}$ with,
in the "little Heston trap" form of Albrecher et al. (2007), which stays on the right branch of the
complex logarithm,

$$\beta = \kappa - \rho\xi iu, \quad d = \sqrt{\beta^2 + \xi^2(iu + u^2)}, \quad g = \frac{\beta - d}{\beta + d},$$
$$C = \frac{\kappa\theta}{\xi^2}\Big[(\beta - d)T - 2\ln\frac{1 - ge^{-dT}}{1 - g}\Big], \qquad D = \frac{\beta - d}{\xi^2}\,\frac{1 - e^{-dT}}{1 - ge^{-dT}}.$$

Calls come from Lewis's (2001) single integral, $C = D_r\big[F - \frac{\sqrt{FK}}{\pi}\int_0^\infty
\mathrm{Re}\big(e^{iux}\phi(u - \tfrac{i}{2})\big)\frac{du}{u^2 + 1/4}\big]$ with $x = \ln(F/K)$, and
puts from parity. The integral is cut off where the integrand drops below $10^{-12}$ (found by probing:
for short maturities the integrand decays like a Gaussian, for long ones only exponentially) and done
with 512-point Gauss–Legendre, once per maturity for all strikes, about 0.4 ms for 120 strikes.

**Checked against:** Fang & Oosterlee's (2008) reference price 5.785155450 (to $10^{-7}$); adaptive
quadrature from 2 days to 2 years (to $10^{-6}$ rupees on a 22,500 forward); Black-Scholes as
$\xi \to 0$ (the gap shrinks like $\xi$ with correlation, like $\xi^2$ without, as it should);
$\phi(-i) = 1$; and a full-truncation Euler Monte Carlo (Lord et al. 2010) within 3 standard errors.

## SABR (`optlab/models/sabr.py`)

$dF = \alpha_t F^\beta dW_1$, $d\alpha_t = \nu\alpha_t dW_2$, $d\langle W_1, W_2\rangle = \rho\,dt$.
Hagan et al.'s (2002) expansion gives the Black implied vol directly. With $z = \frac{\nu}{\alpha}(FK)^{(1-\beta)/2}\ln\frac{F}{K}$
and $x(z) = \ln\frac{\sqrt{1 - 2\rho z + z^2} + z - \rho}{1 - \rho}$,

$$\sigma_B = \frac{\alpha}{(FK)^{\frac{1-\beta}{2}}\big[1 + \frac{(1-\beta)^2}{24}\ln^2\frac{F}{K} + \frac{(1-\beta)^4}{1920}\ln^4\frac{F}{K}\big]}\cdot\frac{z}{x(z)}\cdot\Big[1 + \Big(\frac{(1-\beta)^2\alpha^2}{24(FK)^{1-\beta}} + \frac{\rho\beta\nu\alpha}{4(FK)^{(1-\beta)/2}} + \frac{2 - 3\rho^2}{24}\nu^2\Big)T\Big].$$

$z/x(z) \to 1$ at the money, so near it the code uses $1 - \rho z/2$. $\beta$ is fixed at 1 (lognormal),
which makes $\alpha$ close to the ATM vol: on one day's smile, $\beta$ and $\rho$ are nearly
interchangeable, so fitting both would be noise.

## Calibration (`optlab/calibration.py`, `scripts/calibrate.py`)

Both fits use only quotes between the 5-delta put and call. Beyond them, options trade at a few ticks
at vols no diffusion produces, and a handful of them would dominate any fit.

- **SABR**, per expiry with 5+ quotes: least squares on implied vols over $(\alpha, \rho, \nu)$, from
  four starting points, keeping the best. Median error 0.11–0.22 vol points.
- **Heston**, per day, on expiries from 7 days to a year: residuals are (model − market price) /
  market vega, which is the vol error to first order without inverting implied vols inside the
  optimiser, weighted so each expiry counts equally. Bounded trust-region least squares from three
  fixed starting points plus the previous day's fit, keeping the best. $\kappa$ is capped at 30 (an
  8-day half-life): with only a few monthly expiries, the data can't separate faster mean reversion
  from higher vol of vol, and uncapped fits drift to values like a one-day half-life for small gains.
  Reported error is the RMSE of the model's implied vols against the market's over the fitted quotes.

Fits run in the daily job after ingest and are stored in `models/heston.parquet` and
`models/sabr.parquet` on the `market-data` branch; the app only evaluates them, so a 0.1-CPU Render
instance never has to calibrate anything.
