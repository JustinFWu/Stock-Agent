# Stock-Agent

A research pipeline for a volatility-targeted equity strategy: fetch adjusted
daily bars, forecast realised volatility, and run a cost-aware event-driven
backtest against a same-universe baseline.

It does **not** trade. Nothing here connects to a broker, places an order, or
holds a position. The live path stops at a target weight vector, deliberately —
see `src/strategy/live.py`. A broker interface is Phase 4 of the roadmap, and the
venue chosen there is Alpaca, not Interactive Brokers.

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
| 4 | Broker interface and monitoring | not started |

Phase 3 is built and run, and it failed its pre-committed gate. 12-2 momentum reached
an information ratio of −0.53 against the same-universe equal-weight baseline where the
gate asked for +0.2, and vol targeting reduced risk — volatility 20.4% to 12.1%,
drawdown −26.7% to −15.8% — without improving risk-adjusted return. The working record,
including two defects the run exposed and a demonstrated hole in the gate itself, is in
`docs/phase3-strategy.md`. Phase 4 proceeds regardless: its gate is thirty clean
unattended sessions, an operations test that never depended on having an edge.

Read any absolute performance figure from this repo with the survivorship caveat
attached. The universe is today's large caps, so the names that failed between 2005
and 2026 are missing entirely, and an equal-weight daily-rebalanced run over what
remains scores a Sharpe of about 0.91 with no signal in it at all. `src/data/universe.py` states
the size of that bias and why only same-universe comparisons mean anything.

## Setup

Python 3.12 or newer.

```bash
cd stock-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

API keys are read from the environment (or a `.env` file next to `pipeline.py`)
and are only needed for Phase 4. Nothing in the current pipeline uses them.

## Running it

```bash
cd stock-agent

python pipeline.py --fetch           # Phase 0: download bars for the universe
python pipeline.py --vol-validate    # Phase 1 gate: xgb vs rw / ewma / har-rv
python pipeline.py --train-vol       # fit and save the production forecaster
python pipeline.py --backtest        # Phase 2: the cost-aware backtester
python pipeline.py --build-vol-panel # Phase 3: walk-forward vol forecasts to size against
python pipeline.py --backtest --strategy vol-momentum   # Phase 3: the strategy
```

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

## Tests

```bash
cd stock-agent
python -m pytest tests -q
```

The suite runs on synthetic data and needs no bar cache. The handful of tests that
check the weight paths against *real* bars skip themselves when the cache is empty.
Lint with `ruff check stock-agent/src stock-agent/tests` from the repository root;
the rule set is pinned in `ruff.toml`.

CI runs the same lint, a byte-compile, and the full suite on every pull request.

## Layout

```
stock-agent/
  config.py            every tunable constant, including the risk limits
  pipeline.py          the CLI entry point
  src/
    data/              fetching, the bar cache, the aligned price panel, the universe
    features/          price/volume and realised-volatility features
    labels/            forward-looking targets
    models/            the volatility forecaster and its walk-forward gate
    strategy/          weight formation — one path for backtest and live
    backtest/          the event-driven engine, costs, portfolio accounting, metrics
  tests/
docs/                  roadmap, the Phase 2 and Phase 3 write-ups, and the code review
```
