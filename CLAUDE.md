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
| 4 | Broker interface, reconciliation, the live session path | code in progress — gate 0 / 30 |

346 tests pass, `ruff` clean, `mypy` clean over the gated scope.

Phase 4 delivered so far: the broker vocabulary and Protocol, a crash-capable fake, the
write-ahead order log and recovery, the reconciler, the drawdown kill switch, the pre-trade
veto, the Alpaca adapter and the session runner. Not built: alerting that reaches a human
who is not reading a terminal. The **gate is untouched at zero of thirty sessions**,
because nothing has run against a funded account — "in progress" describes the code, and
the gate describes the operating record. Those are different claims and the table keeps
them apart on purpose.

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
| Drawdown kill switch | `MAX_DRAWDOWN = 0.20` off broker equity, persisted, manual reset — `killswitch.py` |
| Pre-trade veto | post-trade book against every cap above, plus cash and participation — `veto.py` |

Also present: `VOL_SCALE_CAP = 1.5` bounding vol-targeted leverage, and `_check_limits`
rejecting a NaN or negative ceiling — NaN being the dangerous one, since it fails every
comparison and so switches the constraint off while returning a plausible portfolio.

The staleness breaker half-covers: `live.py` refuses a cache older than
`MAX_LIVE_STALENESS_DAYS` and refuses to form weights below `MIN_LIVE_COVERAGE` of the
universe. Both are tested in `test_live_guards.py`. Both are deliberately skipped when an
explicit `as_of` is passed, which is the replay path.

### The gap that was not on the Phase 4 list — closed 2026-09-18

Every limit above **was** enforced only at weight formation — inside `target_weights()`,
once, when the target vector is built. Nothing checked the realised book against them.
That is invisible in a backtest and load-bearing live, because the realised book and the
target diverge by two mechanisms that are there on purpose — `_affordable_scale` scales
buys down when cash is short, and blocked names defer to the next open — plus ordinary
drift between monthly rebalances. A name could be well through 10% with nothing objecting
until the next rebalance date.

`veto.py` is now the second thing: the same caps applied to the **post-trade book computed
from live broker positions**, plus cash sufficiency, participation, and the long-only test
on the post-trade quantity. It rejects and never resizes, and in normal operation it should
never fire — `target_weights()` already returns compliant weights, so a rejection means
something upstream is wrong and is alert-worthy rather than routine.

The weight-formation constraint stays where it is. Both layers are wanted: one shapes the
portfolio, the other refuses to let it be broken.

---

## Phase 4 design sketch

Recorded before implementation so the interface decisions were deliberate rather than
discovered. Every box below now exists except the venue adapter and the runner that walks
the sequence; the text under each is left as written, because the value of the sketch is
that it was written first.

```
  live_target_weights()          EXISTS - target weight vector
          |
          v
  Reconciler                     EXISTS - reconcile.py
          |                      target vs BROKER's positions -> intents
          |                      (never vs our own record)
          v
  Pre-trade veto  <-- KillSwitch  EXISTS - veto.py, killswitch.py
          |                      state persisted, manual reset only
          |                      rejects, never resizes
          v
  Order store                    EXISTS - store.py, recovery.py
          |                      write-ahead; deterministic client_order_id
          v
  Broker (Protocol)  ->  AlpacaBroker EXISTS - alpaca.py   IBKRBroker (later)
                         FakeBroker   EXISTS - fake.py
```

*Built 2026-10-06: `session.py` now walks the sequence and `alpaca.py` is the venue
adapter. The paragraph below is left as it was written, because what was missing and why
is the part that is not recoverable from the diff.*

No session runner yet, so nothing in `stock_agent/` performs the sequence end to end.
`tests/test_execution_chain.py` threads it by hand and is where the claim that the pieces
compose is actually pinned.

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

Relevant if long/short ever arrives: long-only is currently baked into `weights.py:119`
(drops non-positive weights) and `portfolio.py:181-194` (splits sells from buys with
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

**Read:** `docs/roadmap.html` in full, all of `stock_agent/`, `config.py`, `pipeline.py`, the three
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

---

## Session record — 2026-09-15

Built steps 1 and 2 of the Phase 4 sketch above: the broker vocabulary with a crash-capable
fake, then the write-ahead order log and recovery. 164 tests → 224, `ruff` clean. Nothing
outside `stock_agent/execution/` and `tests/` was touched — no existing module changed.

### Build order: deliberately not the sketch's arrow

The diagram is runtime data flow, not build order. Top-down would put the Reconciler first,
against a Broker that does not exist — so its interface gets discovered as an ad-hoc test
stub rather than decided, and the `submit(qty, side)` argument above is worth nothing if the
first Broker-shaped object in the repo is a dict some reconciler test happened to need.

It would also build the cheap parts first. `plan_trades` (`portfolio.py:99`) is already most
of a reconciler — target vs current, a no-trade band, forced exits, `blocked` tracking — and
the veto is a pure function over a post-trade book. Neither can force a rewrite of anything
else. The two things that can are the type vocabulary and the crash-recovery contract, so
those went first, which is what "Build this first" already said.

### Step 1 — `stock_agent/execution/broker.py`, `stock_agent/execution/fake.py`

The vocabulary: `Side`, `OrderIntent`, `Account`, `OrderAck`, `OrderStatus`, `BrokerFill`,
`OrderState`, `BrokerError`, `DuplicateOrderError`, and the `Broker` Protocol as sketched.

- The two `submit` signatures debated above are not in tension: the intent *carries* the
  side, so `submit(intent)` and `submit(ticker, qty, side)` are the same decision. `qty` is
  always positive and direction lives in `Side`.
- `client_order_id` is a **computed property**, `2026-09-15:AAA:<12 hex>`, not a stored
  field — "same session, same intent, same id" has to be structural, not a discipline each
  caller remembers.
- **The digest covers `qty`, so the id alone does not protect a re-sized order.** A recovered
  session that re-sizes a name by one share submits again. Pinning the id to `session:ticker`
  would break legitimate retries after a partial, so the qty stays in and the sketch's
  ordering rule — recovery strictly before generation — is what actually closes it. Recorded
  in the comment on the property so the id is not mistaken for the whole defence.
- `qty` is quantised to `QTY_PRECISION = 6` in `__post_init__`, not only inside the digest,
  so the number hashed and the number sent cannot drift. Non-finite and non-positive are
  refused, NaN being the one that passes `> 0` by failing the comparison.
- `FakeBroker` takes `Fault(method, point, ticker)`. `FailPoint.AFTER_ACCEPT` fires after the
  order is recorded and the book has moved — the window a paper account will not reproduce.
  It is the *remote* broker, so it survives our restart, which is how a restart is simulated.
- The fake does **not** enforce long-only and will short a sell it cannot cover. Faithful to a
  margin account, and it is what makes a later veto test prove something about the veto.

### Step 2 — `stock_agent/execution/store.py`, `stock_agent/execution/recovery.py`

- **Two state machines, kept separate.** `OrderState` is what the venue says; `LocalState`
  (`INTENDED → SUBMITTED → RESOLVED`, plus `ABANDONED`) is what our side knows. The gap
  between them is the whole problem, so they are not one shared enum.
- Append-only JSONL, fsynced per event. A crash during a snapshot rewrite loses the whole
  file at exactly the moment the file is the only thing that knows an order was sent.
- One fold (`_apply`) serves both the live path and the replay, so the in-memory view after a
  write and the view after a restart cannot disagree.
- `submit_intent` writes, then sends. On `BrokerError` it records **nothing** and re-raises:
  the outcome is unknown, and writing "failed" is how a filled order becomes invisible. On
  `DuplicateOrderError` it calls `order_status` rather than assuming either outcome.
- **`recover` is also the fill poll.** Steps 3 and 11 are the same operation, so they are one
  function — two implementations is how the startup path and the end-of-session path come to
  disagree about what a partial means.
- A `None` status is safe only without evidence the broker held it, so `OrderRecord` carries
  `known_to_broker`, set by an ack *or* any status that came back. An acked order the broker
  no longer knows raises: there is no honest inference available.
- A still-working order is neither resolved nor abandoned. `RecoveryReport.blocked_tickers`
  names it and the reconciler decides — the log does not hold policy.
- Two guards beyond the sketch: a torn final line is repaired on open (appending after a
  fragment would splice onto it and destroy a line that parsed before), and `_apply` checks
  each replayed intent still produces the id that was stored. Because the id is derived,
  changing `QTY_PRECISION` or the digest silently re-keys every existing log and the next
  session resubmits the lot; that is now a refusal to start.

### Left open, deliberately

*All three were closed on 2026-09-18; see the session record below. Left as written because
what was open, and why, is the part that is not recoverable from the diff.*

- **Where weights become shares.** `live_target_weights()` returns a weight Series and the
  broker takes `qty`; in the backtest that conversion is inside `plan_trades`, against
  `nav(marks)` with the raw/marks distinction that function is careful about. Live, NAV comes
  from `broker.account().equity`. Nothing in steps 1-2 forces the choice, and it is the
  likeliest source of a silent live/backtest divergence — the exact failure
  `test_weight_parity.py` exists to catch. It is step 3's first decision.
- **"Recovery before generation" is not yet structural.** `recover` reports a working order
  and leaves the response to the caller, so the ordering still lives in a comment until the
  session runner exists. Worth enforcing in code when it does.
- The status table at the top of this file still reads "next — nothing built" for Phase 4.
  Left as-is pending a deliberate call on what "delivered" means for a phase with a
  30-session gate and no session runner yet.

---

## Session record — 2026-09-18

Closed the three items the previous session left open, built the two risk controls the
roadmap's architecture names, and built step 3. 224 tests → 286, `ruff` clean, byte-compile
clean, `--backtest` still runs end to end on real bars.

### Step 3 — `stock_agent/execution/reconcile.py`

**Where weights become shares, decided: NAV is `broker.account().equity`.** The venue is the
one that will settle the trade, and it is the same rule as positions — the broker is the
source of truth, never our own record. Our marks are the estimate, and the two are computed
side by side: `Reconciliation.nav_disagreement` is the signed gap, which feeds the tripwire.
A silent divergence between those two numbers is the failure `test_weight_parity.py` exists
to catch one layer up, so it is measured rather than assumed away.

**The trade arithmetic is `plan_trades`, not a second copy of it.** Target vs current, the
no-trade band, the forced exit of a dropped name, `blocked` tracking — the backtester
already has all of it, and Phase 2's review explicitly named the band as one of the two
surfaces on which backtest and production can diverge. Reusing the function closes the
first of them. The cost was one optional `nav` parameter on `plan_trades` and
`Portfolio.weights`; backtest callers pass nothing and get the identical float they got
before, because there is no external authority to ask in a backtest.

- `_side` derives the side from the existing position, so a buy against a short is
  `BUY_TO_COVER`. A sell that would cross through zero is emitted as a plain `SELL` and
  rejected by the veto on the post-trade quantity — which catches it whatever side it
  claims, and keeps working the day something legitimately shorts.
- Quantisation happens in `_sized`, not in `OrderIntent`: a delta that rounds to zero shares
  has to be *dropped* here rather than raise there.
- `unpriced` is reported rather than inferred from the NAV gap. A holding with no mark is
  valued at zero, so it neither appears in the weights nor objects, and a small one hides
  inside the tolerance.

### "Recovery before generation" is now structural

`reconcile` takes the `OrderStore` and the `RecoveryReport` as required arguments and
refuses to generate anything if the log holds an open order the report does not name. That
is the checkable form of the rule — not "did someone call `recover`", which cannot be
asked, but "is the log in the state a completed recovery leaves behind". It was a comment
since 2026-09-15; it is a `ValueError` now.

### The drawdown kill switch — `stock_agent/execution/killswitch.py`

Three separable parts, as the sketch insisted: a **tripwire** (`evaluate_tripwires`, pure,
returns *every* breach rather than the first), a **state** (on disk, manual reset only), and
an **enforcement point** — which is the veto, in another module. The tripwire and the state
share one call because a tripwire whose result a caller may forget to act on is not a
tripwire; enforcement stays outside because a check that lives inside the thing it checks
cannot be audited.

- `MAX_DRAWDOWN = 0.20`, off broker equity. Derived, not tuned: vol-targeted momentum's
  worst measured drawdown over 17.7 years was −15.8% and the baseline's −32.8%, so 20% sits
  outside what the book being run is known to do and inside what its universe has done.
- The high-water mark is persisted beside the trip state. A peak rebuilt from whatever
  history is on hand is not a peak, and `check` reads it *before* folding in today's
  equity — updating first measures every drawdown as zero.
- First trip wins. The first cause is the diagnosis; whatever fires next is its consequence.
- `reset` requires the name of whoever is clearing it, and nothing in the session path calls
  it. An unattributed reset is how an automatic one gets added later.
- An unreadable state file **refuses to start**. "Assume armed" is the tempting default and
  the wrong one: a kill switch whose state cannot be read is one whose state is unknown, and
  the safe reading of unknown is stopped.

### The pre-trade veto — `stock_agent/execution/veto.py`

The gap flagged at the top of this file, closed. Same caps, applied to the post-trade book
computed from live broker positions, plus cash sufficiency, participation and the long-only
test. Two properties fixed now rather than discovered later:

- **Rejects, never resizes.** This is where the live path deliberately differs from the
  backtester: `_affordable_scale` scales buys down to fit the cash because in a backtest
  that is an accounting convenience. Live, a shortfall means our arithmetic and the venue's
  disagree, and a smaller order hides it.
- **An assertion, not a filter.** `target_weights()` already returns compliant weights, so
  nothing here should ever fire. `is_clean` is the expected answer and `False` is the alert.

Details worth keeping:

- Every check asks whether the intent makes its dimension *worse*, not whether the book is
  compliant afterwards. A sell that reduces an overweight name passes; rejecting it would
  pin the breach in place.
- The long-only check is on the **post-trade quantity**, not on the side. A plain `SELL` of
  more than the book holds opens a short exactly as `SELL_SHORT` does, and only the
  quantity test catches both.
- Gross and sector breaches reject every exposure-increasing intent in the breaching group.
  Blunt on purpose: choosing a subset to drop in order to land just under a ceiling is
  resizing by another name.
- `_increases_exposure` is side-agnostic, because `BUY_TO_COVER` has a positive sign and
  *reduces* the gross book — keying off the side would reject the one order that fixes a
  short.
- An unknown ADV is not a rejection. The cost model treats missing volume pessimistically
  because it is pricing; this layer is refusing, and refusing on absent data stops the book
  on a data gap.

### Documentation corrected

- The Phase 4 status in this file and in `README.md`, the 164 → 286 test count, and
  `stock_agent/execution/` missing from the README's layout tree. The status now separates the two
  things that were being conflated: "in progress" describes the code, and the **gate is
  untouched at zero of thirty sessions** because nothing has run unattended.
- `README.md` claimed "nothing here connects to a broker". Narrowed to what is actually
  true — there is no venue adapter, and the only `Broker` implementation is a test double.
- A stray "I love hot dogs" line at the end of `docs/review.md`.

### Left open, deliberately

- **No session runner and no Alpaca adapter.** The pieces compose — `test_execution_chain.py`
  threads `recover → reconcile → tripwire → veto → submit` by hand and pins it — but nothing
  in `stock_agent/` walks the sequence. That is the next deliberate piece, and it is what steps 3
  and 12 of the session flow are measured by.
- **The cash check assumes same-session settlement of sale proceeds.** True of the margin
  account Alpaca opens, not of a cash account. Counting only pre-trade buying power would
  reject almost every ordinary rebalance, and a veto that fires routinely stops being read.
- **Phase 3 stands as measured.** Nothing here re-runs it, and the demonstrated hole in its
  gate is still recorded rather than patched. A future re-run needs a beta control beside
  the IR *and* point-in-time data; neither is a Phase 4 concern.

---

## Session record — 2026-10-06

Acted on an external code review. Two pieces: the packaging the repo never had, and the
session runner plus venue adapter that the Phase 4 gate was waiting on. 286 tests → 346,
`ruff` clean, `mypy` clean, `--backtest` reproduces its previous numbers exactly.

### What the review got right, and the one thing it understated

The review's ranking was correct and is worth recording because it was external: the
packaging was the only thing in the repo that "looks student-grade", and the gap that
mattered most was that **none of this had ever run**. Both are now addressed.

It understated one item. "Adding mypy or pyright is nearly free given how the code is
written" is true of the execution path and not true of the numeric stack: a full-tree run
reports 93 errors, of which the large majority are `pandas-stubs` imprecision —
`.loc[date]` typed `Series | DataFrame` when the caller knows which, `rolling`/`ewm`
results typed `ndarray` when they are Series. Clearing those means roughly fifty casts
threaded through code that is correct and covered, which trades real risk for a green
check. So the gate is an allowlist, not the whole tree. See below.

### The packaging — `pyproject.toml`, and the flatten

The repo was importable by accident. Forty-two modules and tests each began with
`sys.path.append(Path(__file__).parent...)`, so imports depended on the working directory,
and `ruff.toml` carried a comment explaining that the import-sorting rule was
*unselectable* because no ordering can satisfy a local import that follows a path
mutation. The lint config was apologising for the packaging.

- `stock-agent/` flattened into the repository root; `src/` became `stock_agent/`, a real
  package, with `config.py` and `pipeline.py` inside it. `config.ROOT` moved up one level
  so the bar cache, the saved model and `data/state/` still resolve beside the source tree
  rather than inside the installed package.
- `pip install -e ".[dev]"`. `requirements.txt` is gone and its verified-version record
  lives as a comment in `pyproject.toml`. Two console scripts, which is the next entry.
- All forty-two bootstraps deleted and ruff's `I` rule enabled — the thing the config said
  it wanted and could not have. The only surviving `E402` exemptions are `pipeline.py` and
  the session CLI, where `load_dotenv()` genuinely has to run before `config` reads the
  environment.

Git recorded the moves as renames, so history follows the files.

### The type gate is an allowlist, and it was checked for bite

`files` in `pyproject.toml` names the modules that have been driven to zero, and
`follow_imports = "silent"` is what makes that workable: imported modules are still read
for their real types, so the listed files are checked against the truth rather than
against `Any`, but errors inside an unlisted module are not reported. Without it, adding
one import to the execution path drags the whole numeric stack into the gate, and the only
way to stay green would be to stop checking anything.

Seventeen files are gated and clean. What mypy actually found and fixed, all of it real:
unannotated accumulators in `recover`, a pandas `.items()` key typed `Hashable` handed to
helpers declaring `str` in `reconcile._sized` and three `portfolio` call sites, a manifest
that parsed but need not be an object, a `last_error` narrowed to `ValueError` by its
first assignment, and a `bool(entry) and entry.get(...)` that mypy could not narrow and a
reader could not either.

A green check that cannot fail is worse than no check, so the gate was verified by
breaking it on purpose — a `-> float` returning a string, caught, reverted.

**`ADV_WINDOW` moved from `engine.py` to `config.py`.** The live veto sizes orders against
the same number, and the live execution path importing the backtester to get at a constant
is a dependency pointing the wrong way. `build_strategy` moved out of `pipeline.py` into
`stock_agent/strategy/factory.py` for the same reason: the session CLI needs it, and the
trading entry point depending on the research entry point is backwards.

### `alpaca.py` — the venue adapter

Six methods, each one request, plus the mapping from the venue's words to ours. No SDK:
the Protocol is six calls and the part that has to be right is not the HTTP but the
mapping, which an SDK would hide rather than remove.

- **Paper by default, and the live endpoint needs more than flipping the flag.**
  `AlpacaBroker(paper=False)` raises unless also given
  `i_understand_this_is_real_money=True`. Phase 4's own card says passing its gate is a
  plumbing result and not permission for capital, so the default has to be the one that
  cannot lose money even if every other guard in the repo is wrong.
- **The four sides collapse to two here and nowhere else.** Alpaca takes `buy`/`sell` and
  infers shorting and covering from the position. This is exactly the branch the sketch
  predicted when it argued for keeping `Side` wide: widening the enum stayed cheap because
  the venue mapping is one function rather than a signature everything upstream depends on.
- **Silence is an unknown outcome, not a rejection.** A timeout, a dropped connection or
  unparseable JSON raises `BrokerError` saying so in those words, which is what makes
  `submit_intent` record nothing and re-raise. Writing "failed" is how a filled order
  becomes invisible.
- **A duplicate `client_order_id` is its own error**, matched on the message as well as
  the code, because the numeric code has moved between API revisions and a duplicate
  misread as a generic failure is the one error that would make a recovered session submit
  twice.
- **An unrecognised order status maps to `PENDING` and raises an alert.** Not terminal, so
  recovery keeps asking and the reconciler keeps the name deferred. Mapping an unknown word
  to `FILLED` would have a session size against a position that may not exist; raising
  would halt trading because a venue added a vocabulary word.
- Keys are read from the environment when the adapter is *constructed*, not at import.
  Defaulting them from `config` at import time makes the keys depend on whether
  `load_dotenv()` ran before this module was first imported, which is an import-order bug
  waiting for the one session that imports things in a new order.

The tests stub the transport rather than the network and assert on the mapping, because
that is where a thin adapter can be wrong. The three error-translation cases stub `urlopen`
instead, a level lower, since stubbing `_request` would replace the code under test.

### `session.py` — the runner

The twelve steps as a program. Everything it calls already existed, and
`test_execution_chain.py` already threaded them by hand — which was the honest state of
it, and also the reason the gate stood at zero.

- **`submit=False` is the default and it is not a simulation.** It reads the broker's real
  positions and equity, runs recovery, reconciliation, the tripwires and the veto, and
  stops before `submit()`. Everything that can be wrong about a session is wrong before the
  order goes out.
- **A separate console script**, `stock-agent-session`, not a flag on `stock-agent`. One
  mistyped argument should not separate "re-run the backtest" from "trade the account".
- **One line per session in `data/state/sessions.jsonl`, fsynced.** The gate is thirty
  clean unattended sessions, so the count has to come from a record written as each one
  happens rather than reconstructed from the order log — which only knows about sessions
  that placed an order. A halted or failed session is recorded too; a failure that left no
  trace would be counted as a quiet day.
- **`is_clean` is stricter than "completed".** An alert is by definition the thing that
  needed a human, and unattended is the claim being made. The CLI exits non-zero on
  anything else, which is what a cron line notices.
- **A failed send stops the rest of the queue.** The outcome of that one is unknown, so the
  book is no longer a number this session can size against; carrying on down the list
  would size against it anyway.
- **Deliberate deviation from the sketch.** The sketch puts the staleness and coverage
  guards at step 2, before recovery at step 3. Those guards raise, so a stale cache would
  exit before recovery had settled the previous session's orders. They are now *measured*
  at step 2 and fed to the tripwire, recovery runs next, and the hard refusal stays where
  it already lived, inside `live_target_weights`, which is reached after. Nothing can trade
  on stale data either way; the difference is whether the log gets settled first.
- **Step 12 needed two checks, not one.** The first version compared the broker's closing
  book against the book the session started from plus its own fills. That catches a fill we
  recorded that the venue did not make — and provably cannot catch a position that appeared
  *before* the session started, because the baseline already contains it. Which is the main
  thing step 12 is for. There are now two: an **opening** check against the previous
  session's recorded closing book, plus the orders recovery settled overnight, and the
  **closing** check as before. Each session records its closing book so the next one has a
  baseline; a fresh log has none, so the first session establishes one rather than reporting
  the whole account as drift.

### A real defect the runner found on its first real run

Running a dry session against the actual bar cache, the veto rejected two names of ten on
the per-name cap, both reported as "would reach 10.00% of equity (cap 10%)". Nothing had
breached anything. `target_weights` clips a name to exactly `MAX_WEIGHT`, and the share
count realising 10.000000% of NAV then rounds to the nearest 10⁻⁶ of a share, landing a
few parts per billion either side — CAT at 0.10000000182, MRK at 0.10000000027. A bare `>`
caught the ones that landed above.

That is precisely the failure this layer was designed to avoid. The veto is an assertion
that should never fire, so one that fires on ordinary sessions stops being read, and the
rejections were raised at CRITICAL.

The comparison now carries the resolution of the arithmetic it is checking:
`SHARE_GRID = 10**-QTY_PRECISION`, so the slack on a per-name cap is one share-grid step
valued at the current price over NAV, and on gross and sector caps it is that summed over
the names in the group, because a total over N names carries N names' worth of
quantisation. **This is not a tolerance on the risk limit** — it is derived from
`QTY_PRECISION` rather than tuned, and a real breach is six or more orders of magnitude
larger. Pinned both ways: a position one grid step above the cap passes, and a position one
*share* above it still rejects.

The alternative was flooring the quantity in the reconciler so a buy can never round up
through a cap. That fixes a buy from flat and does nothing about the same boundary reached
by drift, so the check is the right place.

After the fix the same session reports `clean: True` with all ten intents allowed.

### Verification

- 346 tests, no skips, so the real-bar parity tests did run. 56 new: 21 on the runner,
  35 on the adapter, 4 on the cap boundary.
- `ruff check .`, `mypy` and `compileall` clean, each taking its scope from configuration
  so a bare invocation locally is the check CI runs.
- `--backtest --strategy momentum --baseline equal` reproduces 2575.6% total return, CAGR
  16.48%, vol 21.35%, Sharpe 0.73 — identical before and after the refactor, which is what
  makes the `ADV_WINDOW` move and the `portfolio.py` narrowing safe to believe.
- The runner was walked end to end against the real 82-name cache three times in sequence
  — dry run, armed, then a third session on an unchanged book — including the no-op case
  that most of thirty sessions will be.

### Left open, deliberately

- **The gate is still 0 of 30, and that is now the only thing between here and Phase 4.**
  What remains is operational, not code: Alpaca paper keys, a schedule, and thirty
  sessions. Nothing in this session ran against a funded account, paper or live.
- **No alerting that reaches a human who is not at a terminal.** The runner collects
  alerts, records them, and exits non-zero; what it does not do is send anything. That is
  the honest remaining piece of "monitoring and alerting", and it is deliberately last,
  because a notifier built before there is an operating record has nothing to notify about.
- **The cash check still assumes same-session settlement of sale proceeds.** True of the
  margin account Alpaca opens, not of a cash account. Unchanged from 2026-09-18 and the
  reasoning is unchanged.
- **Market orders, `time_in_force=day`.** A session running after the close queues orders
  that fill at the next open, which is why `prices` and `marks` are both that session's
  close and are an estimate for sizing rather than a fill price. Limit orders are a venue
  decision that has not been made and should not be made before thirty sessions say what
  the fills actually look like.
- **`tests/` and the numeric modules are outside the type gate.** Listed in
  `pyproject.toml` with the reason. Worth taking one module at a time, not in one sweep.
- **Phase 3 stands as measured.** Nothing here re-runs it. The demonstrated hole in its
  gate is still recorded rather than patched, and a future re-run needs a beta control
  beside the IR *and* point-in-time data.
