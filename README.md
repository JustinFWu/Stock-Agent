# Stock-Agent

**I built a volatility-targeted momentum strategy, wrote down the threshold it had to
clear before running it, and it missed — an information ratio of −0.53 against a
same-universe baseline where the gate asked for +0.2. This repository publishes that
result instead of tuning the strategy until it passed.**

Deciding not to run that search is the most deliberate thing here. Sharpe sat pinned
near the baseline across every configuration tried, including with the risk caps removed
entirely, so a sweep over the target volatility and the selection fraction would
eventually have manufactured a pass that meant nothing. The negative is the finding. The
same run also exposed a hole in the gate itself — the information ratio is passable by
leverage, demonstrated rather than argued, in
[`docs/phase3-strategy.md`](docs/phase3-strategy.md) §4.3 — and the gate was left
unamended, because that window closed when the first number existed.

What the repo is: a research pipeline for a volatility-targeted equity strategy — fetch
adjusted daily bars, forecast realised volatility, and run a cost-aware event-driven
backtest against a same-universe baseline — plus the execution path that would trade it.

It has **never traded.** As of Phase 4 there is now an Alpaca adapter and a session
runner, so the machinery to place an order exists and has been run end to end against a
fake venue — but no session has run against a funded account, paper or live, so Phase 4's
gate stands at **zero of thirty sessions.** The adapter defaults to the paper endpoint and
the live one needs a second explicit argument on top of flipping the flag; the runner
defaults to a dry run and needs `--submit` to send anything. **Passing Phase 4 would be a
plumbing result and is not permission for live capital** — with Phase 3 failed there is no
measured edge for the plumbing to trade, and that guardrail was written into the phase
card before it became inconvenient.

The full build plan, the pre-committed gates, and the results of each phase are in
[`docs/roadmap.html`](docs/roadmap.html). The Phase 2 backtester and the Phase 3
strategy run have their own write-ups in
[`docs/phase2-backtester.md`](docs/phase2-backtester.md) and
[`docs/phase3-strategy.md`](docs/phase3-strategy.md).

## Where it stands

| Phase | What it delivers | Status |
|---|---|---|
| 0 | ~20y of split- and dividend-adjusted bars for 82 US large caps | done |
| 1 | Volatility forecast, gated against EWMA and HAR-RV on QLIKE and RMSE | gate passed |
| 2 | Cost-aware event-driven backtester; one weight path for backtest and live | gate passed |
| 3 | 12-2 momentum with volatility-targeted sizing | **gate failed** |
| 4 | Broker interface, reconciliation, the live session path | code in progress — **gate 0 / 30 sessions** |

Phase 3 is built and run, and it failed its pre-committed gate. Vol-targeted 12-2
momentum reached an information ratio of −0.53 against the same-universe equal-weight
baseline where the gate asked for +0.2, and vol targeting reduced risk — volatility
20.4% to 12.1%, drawdown −26.7% to −15.8% — without improving risk-adjusted return.
Unscaled 12-2 momentum, the same signal without the sizing, reached +0.14 at a Sharpe
of 0.92 against the baseline's 0.99: also short of the threshold, and short of the
baseline on the plainer measure too. The working record, including two defects the run
exposed and a demonstrated hole in the gate itself, is in
`docs/phase3-strategy.md`. Phase 4 proceeds regardless: its gate is thirty clean
unattended sessions, an operations test that never depended on having an edge.

Phase 4 is under way and its gate is untouched, because nothing has run unattended yet.
The code is complete enough to run: the broker vocabulary and Protocol, a crash-capable
fake, the write-ahead order log and crash recovery, position reconciliation against the
broker's own book, a persisted drawdown kill switch, the pre-trade veto, an Alpaca
adapter, and a session runner that walks the whole sequence and records each session.
Still missing: alerting that reaches a human who is not reading a terminal. The
distinction the status column is drawing is that **"in progress" describes the code and
the gate describes the operating record**, and those are different claims.

Read any absolute performance figure from this repo with the survivorship caveat
attached. The universe is today's large caps, so the names that failed between 2005
and 2026 are missing entirely, and an equal-weight daily-rebalanced run over what
remains scores a Sharpe of about 0.91 with no signal in it at all.
`stock_agent/data/universe.py` states the size of that bias and why only same-universe
comparisons mean anything.

## Setup

Python 3.12 or newer.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

The editable install is what makes `stock_agent` importable; drop `[dev]` to skip the
test and lint toolchain. API keys are read from the environment (or a `.env` file at the
repository root) and are only needed for Phase 4. Nothing in the research pipeline uses
them.

## Running it

```bash
stock-agent --fetch           # Phase 0: download bars for the universe
stock-agent --vol-validate    # Phase 1 gate: xgb vs rw / ewma / har-rv
stock-agent --train-vol       # fit and save the production forecaster
stock-agent --backtest        # Phase 2: the cost-aware backtester
stock-agent --build-vol-panel # Phase 3: walk-forward vol forecasts to size against
stock-agent --backtest --strategy vol-momentum   # Phase 3: the strategy
```

`python -m stock_agent.pipeline` takes the same arguments if you would rather not rely
on the console script being on `PATH`.

`--fetch` first; everything else reads the cached bars and will not download on
its own, so a backtest can never quietly change its own input data mid-run. Add
`--refetch` to replace the bars, or `--rebuild` to recompute derived features.

A vol-targeted run sizes against the walk-forward panel from `--build-vol-panel`, never
the saved production model — that one is fitted on the full history, which is right for
live use and look-ahead inside a backtest. `--baseline` picks what the run is measured
against and defaults to equal weight; the baseline run is not optional, because an
absolute Sharpe on this universe means nothing. Reproducing the Phase 3 gate takes two
runs: `--strategy vol-momentum --baseline equal` for the information ratio, and
`--strategy vol-momentum --baseline momentum` for whether the scaling earns its place.

## Running a session

A separate entry point, deliberately. The research CLI above reads the bar cache and
prints numbers; this one can place orders at a venue, and one mistyped flag should not be
the difference between the two.

```bash
stock-agent-session                      # dry run against the paper account
stock-agent-session --submit             # the same, armed to send orders
stock-agent-session --status             # kill switch, open orders, recent sessions
stock-agent-session --reset-kill-switch "your name"
```

A dry run is the default and it is not a simulation: it reads the broker's real positions
and equity, runs recovery, reconciliation, the tripwires and the veto, and stops before
`submit()`. Everything that can be wrong about a session is wrong before the order goes
out, so the dry run is where you find it.

The sequence is ordered so a crash at any step is recoverable, and the ordering rule that
matters is that **recovery runs before anything is generated** — `reconcile` raises rather
than produce intents if the order log still holds an order the recovery report does not
account for. Each session appends one line to `data/state/sessions.jsonl`, which is the
operating record the thirty-session gate is counted from, and exits non-zero if the
session was anything other than clean.

Alpaca keys come from `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`. The adapter points at the
paper endpoint unless constructed with both `paper=False` and
`i_understand_this_is_real_money=True`.

## Tests

```bash
pytest
```

The suite runs on synthetic data and needs no bar cache. The handful of tests that
check the weight paths against *real* bars skip themselves when the cache is empty.

```bash
ruff check .     # rule set pinned in ruff.toml
mypy             # scope pinned in pyproject.toml
```

Every command takes its scope from configuration rather than arguments, so what runs
locally is what runs in CI. The type check is scoped to `stock_agent/execution` on
purpose and `pyproject.toml` records why: that is the path where a type carries real
weight, and a full-tree run is dominated by `pandas-stubs` imprecision rather than
defects.

CI runs the same lint, type check, byte-compile, and the full suite on every pull
request.

## Layout

`stock_agent/execution/` is where Phase 4 lives, and it reads in dependency order:
`broker.py` is the vocabulary and the Protocol, `fake.py` a crash-capable test double,
`alpaca.py` the only file that knows a vendor exists, `store.py` and `recovery.py` the
write-ahead log and the crash contract, `reconcile.py` target-versus-broker arithmetic,
`killswitch.py` the tripwires and the persisted trip state, `veto.py` the pre-trade
refusal, and `session.py` the runner that walks all of it in order.

```
pyproject.toml         packaging, the test and type-check scopes, dependencies
stock_agent/
  config.py            every tunable constant, including the risk limits
  pipeline.py          the CLI entry point
  data/                fetching, the bar cache, the aligned price panel, the universe
  features/            price/volume and realised-volatility features
  labels/              forward-looking targets
  models/              the volatility forecaster and its walk-forward gate
  strategy/            weight formation — one path for backtest and live
  backtest/            the event-driven engine, costs, portfolio accounting, metrics
  execution/           Phase 4 — the broker interface and the live session path
tests/
docs/                  roadmap, the Phase 2 and Phase 3 write-ups, and the code review
data/                  the bar cache (gitignored; built by --fetch)
data/state/            the order log, the kill switch, the session record (gitignored)
models/saved/          the fitted production forecaster (gitignored)
```
