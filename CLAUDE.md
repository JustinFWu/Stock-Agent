# CLAUDE.md

Working context for this repo: where the project actually stands, the decision that
governs what happens next, and the Phase 4 design sketch that has not been built yet.

The authoritative records remain `docs/roadmap.html` (the plan and the pre-committed
gates), `docs/phase2-backtester.md` and `docs/phase3-strategy.md` (the phase write-ups).
This file is the shorter orientation, plus design work that has no write-up yet because
no code exists for it.

---

## Where the project stands

| Phase | Deliverable | Status |
|---|---|---|
| 0 | ~20y adjusted bars, 82 US large caps | cleared |
| 1 | Volatility forecast vs EWMA / HAR-RV | cleared |
| 2 | Cost-aware event-driven backtester | cleared |
| 3 | 12-2 momentum + vol-targeted sizing | **gate failed** |
| 4 | Broker interface, reconciliation, monitoring | next — nothing built |

164 tests pass, `ruff` clean.

### The governing decision (2026-09-13)

**Finish the entire system first; source alpha separately afterwards.** Phase 3's
failure does not change the plan.

The reason is not optimism about the strategy. It is that **both roads out of Phase 3
run through the same purchase.** Retesting the vol-targeting thesis needs a long/short
book, which needs point-in-time data. Re-running the primary gate honestly needs
point-in-time data. Until that is bought, further alpha work measures the same
survivor-selected sample in new ways. Phase 4 is blocked by none of it, its gate never
depended on having an edge, and it produces the first genuinely out-of-sample evidence
this project has ever had.

**Do not respond to the failed gate by sweeping `vol_target`, `top_fraction`, or
rebalance frequency.** Sharpe sat pinned near the baseline across every configuration
tried, including with the risk caps removed entirely. A search would eventually
manufacture a pass and it would mean nothing. Not doing this is what makes the negatives
on this project worth trusting.

**Passing Phase 4 is a plumbing result and is not permission for live capital.** Two
questions, two bodies of evidence. That guardrail was written into the phase card before
it became inconvenient, which is the only time it can be written honestly.

### What Phase 3 does and does not license

Two failures, with opposite implications — worth keeping separate, because conflating
them makes the negative sound softer than it is:

- **The primary gate failed on its merits.** Unscaled 12-2 momentum reached IR +0.14
  against a +0.2 threshold, at Sharpe 0.92 against the baseline's 0.99. That was a fair
  test of long-only momentum on this universe, run as pre-committed. There is no design
  fix pending for it.
- **The secondary gate — the vol-targeting thesis — was structurally compromised.** The
  thesis is that vol targeting remedies the momentum *crash*, but most of that crash risk
  lives in the short leg and v1 removed the short leg. The remedy had nothing to remedy.
  In 2009 the book did not crash; it *lagged a junk rally* (+9.6% against the universe's
  +41.2%) and vol targeting deepened the lag by sitting in cash. That test was never
  really run.

Survivorship cuts both ways and is the binding constraint on both readings: "momentum has
no edge," measured on names selected for having survived, is close to unfalsifiable —
momentum's would-be losers are disproportionately the names removed before the test began.
The honest claim stays narrow, and `members_asof()` is the seam that makes point-in-time
data a data change rather than a code change.

---

## Risk controls that exist today

Three of the four controls the roadmap's architecture names are built and tested; one
does not exist in any form.

| Control | State |
|---|---|
| Gross / net caps | `MAX_GROSS = 1.0`; long-only enforced by dropping non-positive weights |
| Per-name limit | `MAX_WEIGHT = 0.10`, hard clip |
| Per-sector limit | `MAX_SECTOR_WEIGHT = 0.30`, scaled within the group |
| Drawdown kill switch | **does not exist** — no match anywhere in `src/`, `config.py`, `pipeline.py` |

Also present: `VOL_SCALE_CAP = 1.5` bounding vol-targeted leverage, and `_check_limits`
rejecting a NaN or negative ceiling — NaN being the dangerous one, since it fails every
comparison and so switches the constraint off while returning a plausible portfolio.

The staleness breaker half-covers: `live.py` refuses a cache older than
`MAX_LIVE_STALENESS_DAYS` and refuses to form weights below `MIN_LIVE_COVERAGE` of the
universe. Both are tested in `test_live_guards.py`. Both are deliberately skipped when an
explicit `as_of` is passed, which is the replay path.

### The gap that is not on the Phase 4 list, and should be

Every limit above is enforced at **weight formation** — inside `target_weights()`, once,
when the target vector is built. **Nothing ever checks the realised book against them.**
`engine.py` passes the limits into `target_weights` and never validates the resulting
portfolio.

That is invisible in a backtest and load-bearing live, because the realised book and the
target already diverge by two mechanisms that are there on purpose — `_affordable_scale`
scales buys down when cash is short, and blocked names defer to the next open — plus
ordinary drift between monthly rebalances. A name can be well through 10% with nothing
objecting until the next rebalance date.

So what exists is a **weight-formation constraint**, not the pre-trade veto the
architecture describes as "hard, auditable, and non-negotiable." Phase 4 needs the second
thing.

---

## Phase 4 design sketch

Not built. This is the intended shape, recorded before implementation so the interface
decisions are deliberate rather than discovered.

```
  live_target_weights()          EXISTS - target weight vector
          |
          v
  Reconciler                     target vs BROKER's positions -> intents
          |                      (never vs our own record)
          v
  Pre-trade veto  <-- KillSwitch state (persisted, manual reset only)
          |                      rejects, never resizes
          v
  Order store                    write-ahead; deterministic client_order_id
          |
          v
  Broker (Protocol)  ->  AlpacaBroker      IBKRBroker (later)
```

### Broker

The thin part, and the only part that knows a vendor exists.

```python
class Broker(Protocol):
    def positions(self) -> dict[str, float]: ...          # ticker -> signed shares
    def account(self) -> Account: ...                     # cash, equity, buying power
    def submit(self, intent: OrderIntent) -> OrderAck: ...
    def order_status(self, client_order_id: str) -> OrderStatus | None: ...
    def cancel(self, client_order_id: str) -> None: ...
    def fills(self, since) -> list[BrokerFill]: ...
```

`order_status` and `cancel` are not extras. `order_status` answers "did the order I may
have sent before I crashed actually land?" `cancel` is how the kill switch pulls working
orders instead of merely declining to add new ones.

#### Keep `submit` able to express a short

Nothing in the system generates a short today — `weights.py` drops every non-positive
weight, so the strategy can only say "hold 4% of AAPL," never "be short 4% of AAPL." This
is about the signature, not about behaviour: nothing will short during Phase 4.

With a bare signed share count, "sell 100 AAPL I own" and "short 100 AAPL I do not own"
are the same call, and they are different actions — one needs a locate, margin and borrow:

```python
def submit(self, ticker: str, shares: float): ...          # encodes long-only into the type
def submit(self, ticker: str, qty: float, side: Side): ... # qty positive; vocabulary stays open
```

`Side` is `BUY | SELL | SELL_SHORT | BUY_TO_COVER`, and only the first two are ever passed
today. The point is that the day something generates a short, it is a branch in the Alpaca
adapter rather than a signature change underneath a reconciler, an order store, a veto and
thirty sessions of tests.

**Do not build short support** — no borrow logic, no margin accounting, no locates. The
line is: do not spend an extra keyword to actively assert long-only where staying neutral
is free. Widening the enum later is cheap; changing what the arguments *mean* is not.

Relevant if long/short ever arrives: long-only is currently baked into `weights.py:122`
(drops non-positive weights) and `portfolio.py:175-181` (splits sells from buys with
cash-only affordability logic).

### Reconciliation

One rule: **the broker is the source of truth, never our own record.** If the process died
mid-rebalance, the local record is precisely the thing that is wrong. Every session starts
by asking `broker.positions()` and diffing the target against that.

Idempotency comes from a deterministic id — `f"{session_date}:{ticker}:{intent_hash}"` or
similar. Same session, same intent, same id, so a resubmit is a duplicate the broker
rejects rather than a second position.

The nasty window is between `submit()` returning and the write recording that it was sent.
Two things close it and **both** are needed: write the intent to the store *before*
submitting, and on startup call `order_status` on every intent whose outcome is not
recorded. Ordering matters — **crash recovery runs before any new order is generated**, or
the session doubles up.

### Kill switch

Three separable parts; conflating them is how it fails.

- A **tripwire** evaluating conditions: drawdown off broker equity, staleness, coverage,
  reconciliation mismatch, an unexpected position.
- A **state** that is persisted to disk and requires **manual** reset.
- An **enforcement point**, which is the veto.

The persistence is the part that gets done wrong. A kill switch held in memory clears
itself on restart, and the restart is very often caused by whatever tripped it. Trip →
write to disk → refuse to arm again until a human clears it.

### Pre-trade veto

```python
def veto(intents, positions, account, nav, limits) -> tuple[allowed, rejected]
```

Checked against the **post-trade** book computed from live broker positions: per-name cap,
sector cap, gross cap, no intent that would open a short while v1 is long-only, cash
sufficiency, order size against `MAX_CREDIBLE_PARTICIPATION`, and kill-switch state.

Two properties to fix now rather than later:

- **It rejects, never resizes.** This is the roadmap's own design test — if a limit becomes
  a penalty term inside the optimizer, it has stopped being a risk control and become a
  suggestion. A veto that quietly shrinks an order is negotiating.
- **It is an assertion, not a filter.** `target_weights()` already returns compliant
  weights, so in normal operation the veto should never fire. When it does, something
  upstream is wrong — a partial fill, a deferred name, drift since the last rebalance. A
  rejection is an alert-worthy event, not routine flow control. That is what makes it worth
  having given the caps are nominally already applied.

### Session flow

Ordered so that a crash at any step is recoverable.

1. Load state — if the kill switch is tripped, alert and exit.
2. Fetch bars; staleness and coverage guards (exist, tested).
3. Recover: `order_status` on anything unresolved from the last session.
4. `broker.positions()` — ground truth.
5. `live_target_weights()` (exists).
6. `intents = target - actual`.
7. Tripwire evaluation.
8. Veto.
9. Write-ahead the intents.
10. Submit with deterministic ids.
11. Poll fills, record.
12. Post-session reconcile — assert the broker's book matches intent; mismatch alerts.

Steps 3 and 12 are what the 30-session gate actually measures. Steps 1-2 already exist.

### Build this first

A `FakeBroker` that can be told to die at any step, plus tests that kill it between steps
10 and 11 and assert no duplicate position after restart. The roadmap names a crash
halfway through a rebalance as the real risk rather than signal decay, and it is the one
thing a paper account will not reproduce on demand.

Also keep `members_asof()` intact as the point-in-time seam. Phase 4 is where it is
easiest to break by accident, since a reconciler that assumes a name is always tradable is
the same assumption `engine.py` already documents as a problem the day delisted names
arrive.

### No LLM in the money path

Deterministic checks own the alerts and the kill switch. A model may triage an anomaly,
summarise the session and draft the daily note. It never sizes, sends, or cancels. That
boundary lives in the code, not in intent.

---

## Session record — 2026-09-13

Read the full repo, then made two consistency fixes and produced the Phase 4 sketch above.

**Read:** `docs/roadmap.html` in full, all of `src/`, `config.py`, `pipeline.py`, the three
`docs/*.md` write-ups, and CI. Ran the suite (164 passed) and `ruff` (clean).

**Fixed — stale documentation contradicting the code and the write-ups:**

- `README.md` recorded Phase 3 as "not started" when it was built, run, and failed its
  gate. Now `**gate failed**`, with a paragraph giving the actual result (IR −0.53 against
  +0.2; vol 20.4% → 12.1%, drawdown −26.7% → −15.8%) and stating why Phase 4 proceeds
  regardless — otherwise "gate failed" beside an active Phase 4 reads as a contradiction.
- `README.md` omitted `docs/phase3-strategy.md` from both the doc index and the layout
  tree, and omitted `--build-vol-panel` and `--strategy` from "Running it" — the two flags
  required to reproduce any Phase 3 number. Added, with a note that a vol-targeted run
  sizes against the walk-forward panel rather than the saved model, and that the gate takes
  two runs (`--baseline equal`, then `--baseline momentum`).
- `config.py:50` claimed the truncated book cost "0.19 of information ratio". Both
  `docs/phase3-strategy.md` §4.1 and the roadmap say **0.12** (+0.02 capped against +0.14
  with the floor in place), which is also what the arithmetic gives. Comment-only change;
  `MIN_SELECTED_NAMES` still computes to 10.

**Left alone deliberately:**

- Phase 4 stays "not started" in the README table while no broker code exists, even though
  the roadmap's chip reads *Active*. The roadmap chip describes the phase being worked; the
  README column describes what is delivered.
- The roadmap's "the suite stands at 127" line. It is a point-in-time Phase 2 statement in
  a document that is explicitly living, and its past sections stay as written.
- The Phase 3 gate itself, and its demonstrated hole. Removing the per-name cap pushed IR
  +0.14 → +0.36, through the threshold, while Sharpe stayed pinned at 0.92 and drawdown
  worsened; IR tracked beta. The gate was **not** amended — that window closed when the
  first number existed. Any future re-run needs a beta control beside the IR: an appraisal
  ratio on beta-neutralised active returns, or a joint requirement that Sharpe improve too.
