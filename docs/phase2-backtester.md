# Phase 2 — Cost-aware backtester

**Date:** 2026-08-19
**Branch:** `feat/vol-targeted-momentum`
**Status:** built, tested, verified. Uncommitted.
**Gate:** met, and verified adversarially (see §3).

This is the working record for the Phase 2 session: what was built, what was
measured, what the review found, and what was fixed. It exists because the
reasoning behind several of these decisions is not recoverable from the diff —
particularly the survivorship measurement and the reason the original gate test
was worthless.

---

## 1. What Phase 2 delivered

The roadmap's Phase 2 deliverable: *an event-driven backtest carrying commissions,
spread, and a market-impact estimate, with no-trade bands so that continuously
resizing to hit an exact vol target doesn't churn away the benefit.*

### New modules

| File | Responsibility |
|---|---|
| `src/data/panel.py` | Aligned wide OHLCV panel. `as_of()` truncation is the single look-ahead firewall; `tradable_as_of()` reads eligibility off the price data itself. |
| `src/backtest/costs.py` | Commission, half-spread, square-root market impact. Presets: `ALPACA_COSTS`, `PESSIMISTIC_COSTS`, `ZERO_COSTS`. |
| `src/backtest/portfolio.py` | Cash-and-shares accounting through one mutation point, no-trade bands, cash-constrained execution. |
| `src/backtest/engine.py` | The daily loop and `BacktestResult`. |
| `src/backtest/metrics.py` | Absolute statistics plus `summarize_relative` (excess CAGR, tracking error, information ratio, beta). |
| `src/strategy/weights.py` | `target_weights()` — the shared weight-formation path. Risk limits live here. |
| `src/strategy/live.py` | The live-side caller. Deliberately thin. |
| `tests/` | 47 tests. |

### Changed

- `src/data/universe.py` — rewritten around `UniverseSpec` (see §4).
- `config.py` — portfolio construction limits, shared by backtest and live.
- `pipeline.py` — `--backtest` entry point, always runs a baseline alongside.
- `requirements.txt` — added `pytest`.

### The daily sequence

1. Execute yesterday's decision at today's **open**.
2. Mark to market at today's **close**.
3. If today is a rebalance date, form target weights from data through today's
   close and hold them for tomorrow's open.

The one-day separation between decision and fill is what stops a signal being
traded at a price used to compute it. Costs are charged at execution using volume
and volatility measured strictly before the execution day.

### What the engine does not model

Stated because unstated assumptions are how backtests lie. No intraday fills, no
partial fills or rejects, no borrow costs or shorting, no dividends beyond what
auto-adjusted prices embed, no interest on idle cash unless asked for, no taxes.
Every one of those flatters the result.

Two more that are properties of the data rather than the loop: bars are adjusted
with *today's* split and dividend factors, so the series the engine sees is not
the series a trader in 2006 had; and a position whose bars stop is marked forward
at its last close indefinitely and can never be sold. The second is harmless on a
survivor universe and becomes a real problem the moment delisted names are added.

---

## 2. Current output

`python pipeline.py --backtest --start 2006-01-01`

```
=== Backtest: equal_weight on survivor-82 ===
  period          2006-01-03 -> 2026-08-06  (20.6y)
  total return     1851.7%
  CAGR              15.56%
  ann vol           18.61%
  Sharpe              0.87
  max drawdown      -48.0%
  Calmar              0.32
  ann turnover         17%
  cost drag /yr      0.00%
  costs paid           379   over 186 trading days
  rebalance              M   band 0.50%   max weight 10%
  universe        survivor-82   point-in-time: NO   names 76 -> 82
```

**These numbers are a plumbing check, not a result.** The strategy is
`EqualWeightStrategy`, a parameterless placeholder that exists so the machinery
can be exercised without a signal to tune against it. The roadmap builds the
backtester before the strategy for exactly this reason.

`names 76 -> 82` is the cheap tell for a survivor-only universe: a real universe
loses names over twenty years.

---

## 3. The Phase 2 gate

Pre-committed wording: *one function produces both the backtest weights and the
live weights, enforced by a test asserting byte-identical output for a fixed date.
If the paths can drift, the backtest is fiction.*

**Met.** `target_weights()` in `src/strategy/weights.py` is the only path.
`run_backtest` calls it per rebalance; `live_target_weights` calls it for
production. Risk limits (`MIN_HISTORY_DAYS`, `MAX_WEIGHT`, `MAX_GROSS`,
`NO_TRADE_BAND`) live in `config.py` so neither caller holds its own copy.
`tests/test_weight_parity.py` compares SHA-256 digests of the serialised weights —
literally byte-identical, no tolerance.

### Why the first version of this test was worthless

The review disabled the firewall — replaced `PricePanel.as_of` with the identity
function — and **all four parity tests still passed.** Two independent reasons:

1. `EqualWeightStrategy.propose` returns `1.0/len(candidates)` and never reads
   `history`. The panel it is handed is irrelevant to its output.
2. `PricePanel.tradable_as_of` re-truncates internally, so the candidate list is
   identical either way.

The gate's literal wording was satisfied while the safety property it was written
to buy was untested. That matters precisely at Phase 3, when a history-reading
momentum strategy arrives.

### The fix, and a second trap inside it

Replaced the probe with `HistoryProbeStrategy` — inverse trailing volatility over
63 days, so weights are a sensitive function of exactly which bars are visible.

That alone was still not enough. The test built its "live" panel by calling
`real_panel.as_of(gate_date)` — the very method under test. Sabotaging `as_of`
moved *both* sides together and the digests still matched. Added
`truncate_independently()`, which slices the frames directly, so only the backtest
side relies on the firewall.

### Verification

```
as_of replaced by identity : 2 failed, 2 passed
normal                     : 47 passed
```

The gate now fails when the property it guards is removed.

---

## 4. The survivorship caveat

A subagent was tasked with addressing the standing caveat. It measured the bias
rather than describing it.

### Measurement

2005-01-04 → 2026-08-06, daily rebalanced, dividend-adjusted, no costs:

| | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| Equal-weight, the 82-name universe | 16.84% | 19.1% | 0.91 | −46.9% |
| RSP — equal-weight S&P 500, point-in-time | 10.11% | 20.2% | 0.58 | −59.9% |
| SPY — cap-weight S&P 500, point-in-time | 10.99% | 18.9% | 0.65 | −55.2% |

RSP is the right control: it holds the weighting scheme constant and varies only
whether constituents were chosen with hindsight. **The gap is ~6.7pp/yr of CAGR,
+0.33 of Sharpe, and 13pp of understated drawdown** — available to any long-only
strategy on this universe for free, before any signal. Part of that is a genuine
mega-cap tilt over this window, so treat 6.7pp as an upper bound on pure
survivorship; it is still the correct bound for interpreting a Phase 3 number,
because momentum draws from exactly this pool.

Consistent with the literature: Eisdorfer (JFM 2008) attributes roughly **40% of
momentum profit to delisting returns** specifically — the part this universe
cannot see.

### The design decision: one universe or two?

**Neither a second list nor a rename. One ticker list, two *concepts*, expressed
as a function of date.**

The two sets are currently identical — all 82 names are both live-tradable and in
the backtest. Maintaining two byte-identical lists would be the worst option: it
manufactures the appearance of a point-in-time universe with none of the
substance, doubles the maintenance surface, and forks the very path the Phase 2
gate exists to keep single.

Implemented as `UniverseSpec` in `src/data/universe.py`:

- `members_asof(date)` returns everything for every date, because this universe
  has no membership history. That is the honest behaviour, not a stub — it is the
  seam where a real membership table plugs in without touching the engine, the
  strategy, or the weight path.
- It deliberately does *not* check whether a name had a bar on `date`.
  `PricePanel.tradable_as_of` answers that off the price data, and two places that
  both decide tradability will eventually disagree.
- `caveats` is a field on the universe, not an argument a caller passes, and
  `run_backtest(universe=...)` is required rather than defaulted. A disclosure
  that can be omitted will be omitted.

`ALL_TICKERS` was removed, not aliased — "all" claimed a completeness the list
never had, and an alias keeps the dishonest name alive at call sites.

### What was rejected, and why

**Adding delisted tickers via yfinance.** Tested empirically: of 30 delisted or
acquired S&P names, 23 returned empty and 7 returned *recycled symbols*.

| Symbol | What yfinance serves | Reality |
|---|---|---|
| `FB` | ProShares S&P 500 Dynamic Buffer ETF | not Meta |
| `SPLS` | PIMCO US Stocks PLUS Active Bond ETF | not Staples |
| `STI` | Solidion Technology, nano-cap | not SunTrust |
| `SUNE` | SUNation Energy, first adj. close $2,047,368 | not SunEdison |
| `BBBY` | 5,440 continuous bars 2005→2026, labelled "Bed Bath & Beyond" | Overstock's price path |

`BBBY` is the dangerous one: it passes every sanity check the repo currently has.
This would replace a known, signed, bounded error with an unbounded unknown that
looks like signal.

**Point-in-time membership without the dead names.** Free constituent history
exists (`fja05680/sp500`), but restricting today's 82 names to their membership
dates makes the universe smaller while leaving it survivor-only — and reads as a
fix. Strictly worse than disclosure.

**Treating the listing-date filter as a survivorship fix.** It touches 6 of 82
names (ABBV 2013, META 2012, TSLA 2010, AVGO 2009, V 2008, PM 2008). It removes a
real look-ahead on inclusion and essentially none of the survivorship.

**Haircutting the reported Sharpe.** Once one figure is adjusted, a reader cannot
tell which of the others are measurements. Three real numbers — strategy,
baseline, difference — beat one adjusted number.

### The actual fix, until point-in-time data is bought

Benchmark relatively, inside the biased universe. `summarize_relative` differences
the strategy against a baseline run through the identical engine, panel, costs,
dates and bands. The survivorship premium appears in both series and largely
cancels. `pipeline.py --backtest` always runs the baseline — enforcement by
construction rather than by discipline.

Genuine fix, deferred: Norgate Data US Platinum (~$630/yr) ships delisted
securities *and* historical constituents with symbols disambiguated by last-traded
month. Alternatives: Sharadar SEP, EODHD, Polygon. Beyond fetching, the work is
keying the cache by security identity rather than symbol, and handling delisting
returns (Shumway's −30% for performance-related delistings). **Deferred
deliberately**: spending $630 and two weeks de-biasing a strategy that cannot beat
equal-weight is the wrong order of operations.

---

## 5. Code review findings and fixes

A second subagent reviewed with fresh context and no stake in the code.

### Confirmed clean (verified by instrumentation, not by reading)

- **No look-ahead in the decision path.** Instrumented strategy recording
  `history.dates.max()` on every call: 247 rebalances, **0 violations**.
- **Execution sequencing.** 0 fills landed on a decision date.
- **ADV / volatility lag.** AAPL 2006-02-01: engine ADV `2,503,637,096`, manual
  prior-21-bar mean `2,503,637,096`. The window including the fill day
  (`2,495,696,536`) is not used.
- **Cash/share reconciliation.** NAV rebuilt independently from the fill log
  across 5,180 days and 512 fills: worst relative error **9.3e-16**.
- **Slippage charged exactly once.** Cash paid on buys minus ref notional =
  `182.89587177`; reported spread+impact = `182.89587177`.
- **Square-root impact law.** 4× trade → 2× impact; `0.5 × 0.02 × √0.01 = 10bp`.
- **Panel integrity.** No duplicate dates, monotonic index, no non-positive
  prices, zero cells with a valid close but a NaN open.

### Major findings — all fixed

**1. The gate test could not detect its own firewall's removal.**
Covered in §3.

**2. Buy scaling permitted silent, unpriced margin.**
`execute()` measured buying power against *ref-price notional*, ignoring slippage
and commission — which is precisely what the shortfall consists of. The 0.999
headroom was consumed by any cost model above ~10bp.

Confirmed in the full engine (20.6y, thinned volume, `PESSIMISTIC_COSTS`):
525 days of negative cash, max gross **1.0234** against a 1.0 ceiling, borrowing
up to 2.34% of NAV financed at 0%. It flattered exactly the run meant to be the
stress test.

*Fix:* scale against cost-inclusive required cash. Conservative by construction —
scaling a trade down reduces its impact, so realised cost is always at or below
the amount reserved.

*Verified:* same stress case now gives **0 negative-cash days, min cash 0.00, max
gross exactly 1.0000.**

The test that was supposed to catch this (`test_buys_are_scaled_to_available_cash`,
docstring *"must never quietly borrow to pay its own costs"*) ran with
`ZERO_COSTS` — with no costs there are no costs to borrow for. Now uses a
pessimistic model on a thin, volatile name.

**3. Turnover and cost drag biased low by the NAV denominator.**
Both divided by `equity.mean()`. On a curve that compounds 18×, early-period
trading is divided by a NAV several times its actual size. Measured on the live
20.6y run: turnover reported **14.32%** against a path-consistent **16.97%**;
cost drag **0.0033%** against **0.0039%**. Both understated ~18%, in the
flattering direction, in the two statistics that exist *because* they are where a
plausible-looking strategy fails.

*Fix:* accumulate each day's flow against that day's NAV.

**4. A NaN daily volatility silently zeroed market impact.**
`fallback_participation` established the principle that missing volume must not
read as a free trade — but the volatility leg had no equivalent. Worse, the two
are structurally coupled: ADV and daily vol derive from the same bars over the
same rolling window, so **whenever ADV is NaN, vol is NaN too**, the impact term
short-circuits to zero, and the pessimistic participation fallback is multiplied
by nothing. The fallback the module is built around was unreachable in engine use.

*Fix:* `fallback_daily_vol = 0.03`, and a distinction between an *unknown*
volatility (use the fallback) and a *measured* zero (no impact — correct).

**5. A missing open bar collapsed NAV and triggered a portfolio-wide sell.**
The engine marks with ffilled closes so a halted name doesn't print a fake round
trip, but handed `plan_trades` the *raw* opens, where `_price_or_zero` values an
unpriced holding at zero. Demonstrated: NAV 20,000 → 10,000 with one name halted,
producing a −50 share sell of the healthy position that was already at target.
Fake loss, fake recovery next day, real costs on a trade that should never have
been placed.

*Fix:* `plan_trades` takes raw prices (tradability) and marks (valuation)
separately. Engine supplies `open_marks = opens.combine_first(marks)`.

### A sixth, found while fixing #1

**`_apply_constraints` clipped before scaling.** Any strategy proposing conviction
scores rather than portfolio fractions — raw inverse-volatility, for instance —
has every value above `max_weight`, so every name clipped to exactly the cap and
the subsequent gross rescaling returned a **perfectly equal-weight portfolio**.
The signal deleted, silently, with entirely plausible-looking output.

This is exactly what Phase 3's vol-targeted sizing would have hit. Fixed the
ordering: scale to the gross ceiling first, so relative shape survives; the cap
then binds only on genuinely oversized names. `tests/test_weights.py` guards it.

### Minor findings fixed

- `n_trades` → `n_trade_days` (it counted trading days, not fills).
- `panel.py` off-by-one: `min_history` now counts bars strictly *prior* to the
  date, matching its docstring. Relevant because a 12-2 formation window reaches
  back thirteen months.
- Oversized-participation check is now `>=`, so a fill with unknown ADV — assigned
  exactly the fallback participation — actually raises the caveat.
- A window containing no rebalance produced a silent, vacuous `0.0%` return. Now
  raises a caveat.
- Negative cash, should it ever occur, now raises a caveat.
- `test_gross_exposure_never_exceeds_the_ceiling` asserted `< 0.75` against a 0.60
  ceiling; now asserts exactly on rebalance days.
- `test_cash_rate_compounds_when_uninvested` asserted only `end > start`; now
  checks the compounding exactly.

### Minor findings noted but not changed

- `NO_TRADE_BAND = 0.005` is an *absolute* weight band. Across 82 equal-weighted
  names (1.22% each) a name must drift 41% of its own size before it trades. Fine
  at Phase 2; **a band proportional to target weight would behave consistently
  across portfolio widths**, and Phase 3's continuous vol resizing is exactly the
  churn the band exists to damp. Worth revisiting then.
- The gate covers *weights given a date*. The **rebalance schedule** and the
  **no-trade band** live only in the engine and are the two remaining surfaces on
  which backtest and production can diverge. Phase 4 will have to reimplement
  both; better to move them behind the shared path before that, not after.
- Negative proposals are silently dropped. Documented in the `Strategy` protocol
  and consistent with long-only `MAX_GROSS`, but silence is a weak default for a
  risk layer.

### Hygiene pass

- `FILL_COLUMNS` declared once, so the empty and populated fill frames cannot
  describe different schemas — a difference that would only surface on a run that
  traded nothing.
- `PricePanel` reconstruction by keyword rather than positional splat. A
  positional rebuild silently depended on field declaration order; reordering the
  dataclass would have swapped opens for highs with no error.
- `_fill` → `_book_fill` (it mutates, and the old name did not say so).
- `summarize` / `summarize_relative`, `format_summary` /
  `format_relative_summary` — consistent pairs.
- `rows` → `daily_rows`, `oversized` → `oversized_fills`.
- Removed an unused fixture parameter and an unused import; pyflakes clean.
- The measured survivorship figures are stated once, in `SURVIVORSHIP_CAVEAT`.

---

## 6. Verification

```
pytest tests -q                      47 passed
pyflakes (src, tests, pipeline)      clean
pipeline.py --backtest               runs end to end, 20.6y
as_of sabotage                       2 parity tests fail (correct)
margin stress case                   0 negative-cash days, gross 1.0000
```

Nothing was committed.

---

## 7. Open decision: the Phase 3 gate

**The gate as written is unfalsifiable, and this needs a decision before any
Phase 3 number exists.**

The roadmap says: *net-of-cost Sharpe above ~0.4 across the full history.* On this
universe that is cleared at **0.87** by equal-weighting all 82 names and never
trading. Even honest point-in-time RSP clears it at **0.58**. The gate cannot fail.

There is a second, independent problem with it: 0.3–0.5 is the **long/short**
figure from the momentum literature, whose benchmark is zero. It has been attached
to a **long-only** strategy, whose benchmark is the universe. Those are not
comparable quantities.

Recommended replacement — to be written into `roadmap.html` *before* Phase 3 runs:

1. **Primary, bias-cancelling:** information ratio of
   `strategy − EqualWeightStrategy`, both net of costs on the identical panel,
   **> 0.2**. This is the gate.
2. **Secondary, kept as designed:** vol-targeted must measurably beat unscaled
   momentum.
3. **Absolute floor, if kept at all:** set at the measured equal-weight baseline's
   net Sharpe — about **0.9** here, not 0.4 — and labelled "necessary, not
   sufficient."

Raising 0.4 to some larger number is not recommended: the bias magnitude has an
error bar wider than any increment worth picking, and an absolute threshold stays
wrong in one direction or the other depending on the decade.

The machinery this needs is built (`summarize_relative`, and `--backtest` always
running the baseline). **`docs/roadmap.html` was deliberately not edited** —
changing a pre-committed gate is the project owner's call, and the honest window
for it is now, before any Phase 3 result exists.

Also pre-commit, per the subagent: *a passing Phase 3 is provisional until re-run
on point-in-time data; the Phase 4 paper record is the first genuinely
out-of-sample evidence this project will have.* The roadmap already says the
second half — treat it as binding.

---

## 8. Notes for Phase 3

- Raise `MIN_HISTORY_DAYS` to ~273 (13 months). A 12-2 formation window reaches
  back thirteen months, so a name with exactly 252 bars would rank on a truncated,
  higher-volatility window against names measured over a full one.
- Set `cash_annual_rate` deliberately. It defaults to 0, which understates any
  strategy holding meaningful cash — and vol targeting will hold a lot.
- Do not re-open Phase 1. The vol-forecast gate is a *relative* forecast-error
  comparison in which all four models score identical rows in the identical
  universe; survivorship does not contaminate the ranking. It does mean absolute
  vol levels are understated — the universe excludes names that had terminal
  distress — which matters for calibrating the vol *target*, not for the verdict.
- Return prediction remains a settled dead end.
