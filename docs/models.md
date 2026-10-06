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

## Variance swaps and India VIX (`optlab/variance.py`)

A variance swap's fair strike is replicated by out-of-the-money options weighted by $1/K^2$:
$\sigma^2_{VS}T = 2e^{rT}\big[\int_0^F P(K)K^{-2}dK + \int_F^\infty C(K)K^{-2}dK\big]$, discretised
like the VIX (each strike weighted by half the distance to its neighbours). Interpolating total
variance to 30 days reproduces India VIX: over 562 days, correlation 0.979, 0.30 points below it on
average (the strip stops at the quoted strikes), 0.45 points off on a typical day.

## Rough Bergomi (`optlab/models/rough_bergomi.py`)

$$V_t = \xi_0(t)\exp\Big(\eta Y_t - \tfrac{\eta^2}{2}t^{2H}\Big), \qquad Y_t = \sqrt{2H}\int_0^t (t - s)^{H - 1/2}dW_s, \qquad \frac{dS_t}{S_t} = \sqrt{V_t}\,dB_t, \quad B = \rho W + \sqrt{1 - \rho^2}W^\perp$$

(Bayer, Friz and Gatheral, 2016). $Y$ is a Riemann–Liouville fractional Brownian motion with
$\mathrm{Var}(Y_t) = t^{2H}$, so $E[V_t] = \xi_0(t)$, the forward variance curve. For $H < 1/2$ its
paths are rougher than Brownian motion's, and the at-the-money skew behaves like $T^{H - 1/2}$ as
$T \to 0$; a Markovian model like Heston has a skew that levels off instead.

**Simulation.** The hybrid scheme with $\kappa = 1$ (Bennedsen, Lunde and Pakkanen, 2017; as in
McCrickerd and Pakkanen, 2018): on a grid of 6-hour steps, the integral over the most recent step is
drawn exactly, jointly with that step's Brownian increment (a 2×2 covariance, factored by hand
because it's singular at $H = 1/2$); the rest is a Riemann sum with the kernel evaluated at the
optimal points $b_k = \big(\frac{k^{a+1} - (k-1)^{a+1}}{a+1}\big)^{1/a}$, $a = H - 1/2$, done for every
path at once as an FFT convolution. The spot is simulated relative to its forward, $X = S/F$, and the
terminal $X$ is rescaled to mean exactly 1 so put-call parity holds exactly. 20,000 paths over 91 days
take about 0.25 s.

**Checks:** $E[X_T] = 1$; $E[V_t] = \xi_0(t)$ for a stepped curve; $\mathrm{Var}(Y_t) = t^{2H}$ for
$H = 0.05, 0.1, 0.3$; at $H = 1/2$, $Y$ is exactly the Brownian motion; the simulated ATM skew falls
with slope −0.35 between 3 and 28 days for $H = 0.1$ (theory −0.4 as $T \to 0$) and stays flat for
$H = 1/2$.

**Calibration** (`calibration.fit_rough_bergomi`), NIFTY expiries from 2 to 91 days between the
5-delta wings: $H$, $\eta$, $\rho$ and the forward variance curve (one level per interval between
expiries, started from the ATM term structure), by least squares on vega-weighted price errors.
Every evaluation uses the same 20,000 antithetic paths (the same random numbers on every day too),
so the objective is smooth and a small parameter change moves prices far less than Monte Carlo noise
would. The Volterra process is cached while $H$ is unchanged and the variance factor while $H$ and
$\eta$ are, so most of the optimiser's finite-difference steps skip the FFT. About 10 s a day.

Why fit $\xi_0$ rather than take it from variance swaps: the replicated variance swaps were too noisy
expiry by expiry. A one-week strip includes the lottery-ticket wings (9.7% against 7.6% at the money
on 23 Dec 2025), and a thinly quoted expiry's strip stops short (7.8% against 8.8%). Taking them as
given put whole expiries 1.3 points off.

**Comparing with Heston fairly.** Heston is refitted each day to the same quotes. Rough Bergomi has
an extra level per expiry, so its overall error isn't directly comparable; the *shape error* (RMSE
after removing each expiry's average miss) compares how well each gets the smile's shape, which is
what roughness is about.

## Exotic options (`optlab/exotics.py`)

Asian (arithmetic and geometric averages over the fixings), barrier (up/down, in/out) and
floating-strike lookback options, on paths from Black-Scholes, Heston or rough Bergomi.

**Continuous monitoring from daily fixings.** Between fixings the log-price is treated as a Brownian
bridge with that interval's variance (exact under Black-Scholes; the interval's realised variance
under stochastic volatility). A knock-out pays the vanilla payoff times the product over intervals of
the bridge's survival probability $1 - e^{-2(x_0 - b)(x_1 - b)/w}$, a conditional expectation that's
smoother than sampling crossings; a knock-in pays the rest. Lookback extremes are sampled exactly
from each interval's bridge: $\min = \frac{1}{2}\big(x_0 + x_1 - \sqrt{(x_1 - x_0)^2 - 2w\ln U}\big)$.

**Closed forms under Black-Scholes** check the Monte Carlo: the discretely monitored geometric Asian
(the log of the average is normal), the eight standard barriers (Haug, 2007), and Goldman–Sosin–Gatto
floating lookbacks. Every one matches within Monte Carlo error, and knock-in + knock-out = vanilla
exactly.

**numba.** For Black-Scholes and Heston, compiled kernels simulate each path and reduce it on the fly
(running average, extremes, survival probability) without storing it: for 200,000 paths and 50 fixings
0.8 s and no path arrays, against 1.1 s and 162 MB in numpy. Rough Bergomi needs each path's whole
history (the Volterra integral), so it stays in numpy with an FFT. numba is imported only when an
exotic is priced.

**Variance reduction** (arithmetic Asian, Black-Scholes), measured as variance per path against plain
Monte Carlo: antithetic paths about 2×; the geometric Asian as a control variate (closed form, and it
moves almost in lockstep with the arithmetic average) about 1,800×; scrambled Sobol points with a
principal-component path construction about 7,600×; both together about 200,000×.
