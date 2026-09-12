# Phase 3 — Signal, sizing, and the risk layer

**Date:** 2026-09-12
**Branch:** `feat/vol-targeted-momentum`
**Status:** built, tested, run. **Gate failed.**
**Verdict:** 12-2 momentum has no risk-adjusted edge on this universe, at any
setting tried. Vol targeting works as risk control and does not earn its cost here.

This is the working record: what was built, what the gate returned, two defects the
run exposed, and what the result does and does not license anyone to conclude.

---

## 1. The gate

Pre-committed in `roadmap.html` on 2026-09-12, before any Phase 3 number existed.

| Test | Threshold | Result | |
|---|---|---|---|
| **Primary** — information ratio vs same-universe equal weight, net of costs | > 0.2 | **−0.53** | **FAIL** |
| **Secondary** — vol-targeted beats unscaled momentum | must improve | Sharpe 0.91 vs 0.92, IR **−0.81** | **FAIL** |
| **Floor** — net Sharpe | ≥ 0.87 | 0.91 | pass (necessary, not sufficient) |

17.7 years, 2008-11-12 to 2026-08-06, monthly rebalance, Alpaca costs, cash at 2%.
The window starts where the out-of-sample vol panel starts; see §3.

```
                        CAGR    ann vol   max DD   Sharpe   beta   IR vs equal   gross
equal weight           19.20%   17.29%   -32.8%    0.99     —          —         94.6%
momentum               20.56%   20.36%   -26.7%    0.92    0.95      +0.14       88.1%
vol-targeted momentum  13.04%   12.12%   -15.8%    0.91    0.53      -0.53       61.0%
```

Every Sharpe above is netted of the 2% cash rate. Neither arm beats the baseline's
0.99. That is the result in one line.

---

## 2. What the two findings actually are

**The signal is the problem, not the sizing.** Unscaled 12-2 momentum reaches an
information ratio of +0.14 against equal weight, below the 0.2 threshold, at a
*lower* Sharpe than the baseline (0.92 against 0.99). Every downstream number
inherits this. Vol targeting cannot rescue a signal that is not there.

**Vol targeting did exactly what it is designed to do, and it was not worth it.**
Volatility 20.4% → 12.1%, max drawdown −26.7% → −15.8%, beta 0.95 → 0.53. The
sizing layer is not broken; the mean target gross is 55%, it never reaches the
ceiling, and realised vol lands at 12.1% against a 10% target. But holding the
cash that requires cost 6.2pp/yr of CAGR across a 17-year bull market, and the
risk-adjusted return did not improve.

Both forecast sources fail, and the model panel is marginally better than the
parameter-free baseline — IR −0.53 for the XGBoost panel against −0.54 for EWMA —
which is consistent with Phase 1's 5–9% forecast edge being real and small. The
failure is not an artefact of the forecast.

### The thesis was never really testable here

The roadmap's thesis is that vol targeting is the documented remedy for the
documented *momentum crash*. Episode returns, vol-targeted against unscaled:

| Episode | equal | momentum | vol-targeted | protected? |
|---|---|---|---|---|
| 2009 momentum crash (Mar–Jun) | +41.2% | +9.6% | +4.1% | **no** |
| 2011 Aug selloff | −5.2% | −12.0% | −6.3% | yes |
| 2015–16 correction | −3.7% | +0.1% | +1.0% | yes |
| 2018 Q4 | −11.8% | −20.1% | −14.5% | yes |
| 2020 covid crash | −12.7% | −1.7% | −5.2% | **no** |
| 2022 bear | −11.5% | +4.7% | +4.8% | flat |

Three of six, and it failed in the single episode the thesis was built on. In
2009 the long-only book did not crash — it returned +9.6% while the universe
returned +41.2%. It *lagged a junk rally*, and vol targeting deepened the lag by
sitting in cash through it.

This is not a bug, and the roadmap already contains the reason. The decision
*"Long-only for v1"* states that most of momentum's crash risk lives in the short
leg. v1 removed the short leg, which removed most of the risk the sizing layer
was built to neutralise. **Thesis and implementation have been in tension since
the design; this is the first measurement of it.** Any future reading of the
vol-targeting thesis needs a long/short book to be a fair test.

---

## 3. Look-ahead: why the backtest does not use the saved model

`train_vol_model` fits on the full history. That is correct for live use and
look-ahead inside a backtest — a 2010 position sized by a model that has seen
2020 is not a measurement.

`build_oos_vol_panel` produces the forecast the backtest sizes against: each row
comes from a model trained only on dates strictly before its own test block,
reusing the `split_frames` purge the Phase 1 gate already validates. Five folds
over 430,185 rows yields 4,450 dates × 82 tickers, starting 2008-11-12 because
the first fold is training-only. That start date is why the backtest window
begins there rather than in 2005.

`test_a_forecast_never_sees_the_labels_of_its_own_block` is the guard, and it was
verified adversarially: `split_frames` was sabotaged to train on its own test
block, the test failed, and it passed again on restore. A test for a look-ahead
property that has not been shown to fail under look-ahead is not evidence.

---

## 4. Two defects the run exposed

### 4.1 The per-name cap silently truncated the book — fixed

A top decile of 82 names is 8 names. Under a 10% per-name cap, 8 names top out at
80% gross, so the risk layer quietly converted the remaining 20% into cash — and
`proposal_scale = "absolute"` then read that leftover as a *deliberate* cash
position the strategy never chose. Measured, this cost momentum 0.12 of
information ratio (+0.02 capped against +0.14 with the floor in place).

This is precisely the class of bug the `proposal_scale` contract exists to
prevent, and it slipped through anyway, because the contract governs what the
*strategy* means and says nothing about what the *cap* then does to it.

Fixed structurally rather than by retuning a number:

```python
MIN_SELECTED_NAMES = math.ceil(MAX_GROSS / MAX_WEIGHT)
```

The selection floor is derived from the risk limits, so a book too concentrated
to be held at full investment can no longer be proposed. Raising `MAX_WEIGHT`,
lowering `MAX_GROSS`, or changing the decile all move the floor with it.

### 4.2 One report carried two Sharpe conventions — fixed

`summarize` netted the risk-free rate; `summarize_relative` did not. The same
curve scored 0.97 in one block of the output and 1.14 in the other, and the gate
is written as "net Sharpe above X".

**This one had teeth, because it flattered exactly the wrong strategy.** The
vol-targeted book holds ~39% cash against a baseline that holds almost none. Not
charging that cash the risk-free rate inflated its Sharpe relative to the
baseline, and fixing the convention reversed the comparison: an apparent
1.14-against-1.10 *win* became a 0.91-against-0.99 *loss*. The secondary gate's
sign depended on a reporting inconsistency.

`summarize_relative` now takes `risk_free_rate` and applies the same definition,
and `test_both_summaries_report_the_same_sharpe` pins them together. The
information ratio is unaffected by construction — a rate common to both series
cancels in the active return — and there is a test for that too.

---

## 5. The gate itself has a hole

**The information ratio is passable by leverage, and this was demonstrated rather
than reasoned about.** Relaxing the per-name and sector caps:

| caps | gross | beta | **IR** | Sharpe | baseline Sharpe | max DD |
|---|---|---|---|---|---|---|
| 10% / 30% (as gated) | 88.1% | 0.95 | +0.14 | 0.92 | 0.99 | −26.7% |
| none | 99.5% | 1.10 | **+0.36** | 0.92 | 0.99 | −31.9% |

IR rises with beta while Sharpe stays pinned at 0.92 and drawdown gets worse. The
baseline is a fixed equal-weight book, so any higher-beta portfolio earns positive
active return mechanically in a rising market. A gate written only as IR > 0.2 can
be cleared by buying more of the same risk.

**The gate was not amended.** That window closed the moment the first number
existed, and the failure stands on its own terms. But a future Phase 3 gate needs
a beta control alongside the IR — the appraisal ratio on beta-neutralised active
returns, or a requirement that Sharpe improve as well. Recorded here so whoever
re-runs this on point-in-time data does not inherit the hole.

---

## 6. Verification

```
pytest tests -q                      164 passed
ruff check src tests                 clean
look-ahead guard under sabotage      fails correctly, passes on restore
vol-targeted mean gross              55.2%, never at the ceiling, range 18.9%-77.6%
realised vol vs 10% target           12.1%
```

---

## 7. What this result does not license

The universe is 82 survivors. **"Momentum has no edge" measured on names selected
for having survived is close to unfalsifiable** — momentum's would-be losers are
disproportionately the names that were removed before the test began, which biases
this specific test in a direction that is easy to state and impossible to size
from inside the sample.

So the honest claim is narrow: *on this survivor-selected universe, over this
window, 12-2 momentum did not beat its own baseline on a risk-adjusted basis, and
vol targeting reduced risk without improving risk-adjusted return.* It is not a
finding about momentum.

What would make it one is point-in-time constituents with delisted histories —
see `phase2-backtester.md` §7 for the options and the `members_asof()` seam that
makes it a data change rather than a code change. That is the one remaining
experiment that could change the answer rather than the number.

**Not worth doing:** sweeping `vol_target`, `top_fraction`, or rebalance
frequency. The shape of the result is now known — Sharpe pinned near the baseline
across every configuration tried. A parameter search would eventually produce a
pass and it would mean nothing. That is the trap the cross-sectional ranking
branch was killed to avoid.
