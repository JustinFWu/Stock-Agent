"""
The event-driven backtest loop.

Phase 2's deliverable. It is built before the strategy on purpose: a backtester
written after the signal tends to grow, one convenience at a time, into a machine
for confirming the signal.

The daily sequence, and the reason for each step:

  1. Execute yesterday's decision at today's OPEN. Decisions are made after a
     close and filled at the next open, so a signal can never be traded at a
     price that was used to compute it. This single day of separation is the
     difference between a backtest and a fantasy.
  2. Mark to market at today's CLOSE. NAV, weights, drift.
  3. If today is a rebalance date, form target weights from data through today's
     close and hold them for tomorrow's open.

Costs are charged at execution using volume and volatility measured strictly
before the execution day, so the fill is priced with information the trader
actually had.

What this engine does not model, stated plainly because unstated assumptions are
how backtests lie: no intraday fills, no partial fills or rejects, no borrow
costs or shorting, no dividends beyond what auto-adjusted prices already embed,
no interest on idle cash unless asked for, and no taxes. Every one of those makes
the reported result better than reality rather than worse.

Two more that are properties of the data rather than of the loop. Bars are
split- and dividend-adjusted with today's factors, so the volatility and returns
the engine sees are not the series a trader in 2006 actually had — unavoidable
with this source, and standard, but it is still an assumption. And a position in
a name whose bars stop is marked forward at its last close indefinitely and can
never be sold, because there is no price to sell it at; that is harmless on a
universe of survivors and becomes a real problem the moment delisted names are
added, which is where the roadmap says the universe must eventually go.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MAX_GROSS, MAX_WEIGHT, MIN_HISTORY_DAYS, NO_TRADE_BAND, TRADING_DAYS
from src.backtest.costs import ALPACA_COSTS, MAX_CREDIBLE_PARTICIPATION, CostModel
from src.backtest.metrics import format_summary, summarize
from src.backtest.portfolio import Portfolio, execute, plan_trades
from src.data.panel import PricePanel
from src.data.universe import UniverseSpec
from src.strategy.weights import Strategy, target_weights

# Windows for the liquidity and risk inputs the cost model needs. Both are lagged
# by a day before use so an execution is never priced with its own day's data.
ADV_WINDOW = 21
VOL_WINDOW = 21

# Maps the schedule the caller asks for to a pandas resample rule. Spelled out
# rather than passed through because pandas 3 removed the bare "M"/"Q" aliases,
# and a rebalance frequency is not something to leave to a library rename.
REBALANCE_RULES = {"D": None, "W": "W", "M": "ME", "Q": "QE"}


@dataclass
class BacktestResult:
    """
    Everything a run produced, including what is wrong with it.

    `caveats` is not decoration. A Sharpe from this engine is measured over a
    universe of names that are large caps *today*, which is the one bias no
    amount of careful accounting inside the loop can remove. Carrying the caveat
    on the result object means it travels with the number into any report, and
    `describe` prints it directly under the headline figures.
    """

    equity: pd.Series
    daily: pd.DataFrame
    fills: pd.DataFrame
    targets: pd.DataFrame
    metrics: dict
    meta: dict
    caveats: list[str] = field(default_factory=list)

    def describe(self) -> str:
        pit = "yes" if self.meta["universe_point_in_time"] else "NO"
        lines = [
            f"\n=== Backtest: {self.meta['strategy']} on {self.meta['universe_id']} ===",
            format_summary(self.metrics),
            f"  rebalance       {self.meta['rebalance']:>8}   band {self.meta['no_trade_band']:.2%}"
            f"   max weight {self.meta['max_weight']:.0%}",
            # A universe that never loses a name across twenty years is survivor-only
            # by construction. Printing both counts says so without anyone having to
            # write the sentence.
            f"  universe        {self.meta['universe_id']}   point-in-time: {pit}"
            f"   names {self.meta['n_names_first']} -> {self.meta['n_names_last']}",
        ]
        if self.caveats:
            lines.append("\n  Read the above with these in mind:")
            lines += [f"    - {c}" for c in self.caveats]
        return "\n".join(lines)


def run_backtest(
    panel: PricePanel,
    strategy: Strategy,
    *,
    universe: UniverseSpec,
    start=None,
    end=None,
    rebalance: str = "M",
    initial_cash: float = 100_000.0,
    costs: CostModel = ALPACA_COSTS,
    no_trade_band: float = NO_TRADE_BAND,
    min_history: int = MIN_HISTORY_DAYS,
    max_weight: float = MAX_WEIGHT,
    max_gross: float = MAX_GROSS,
    cash_annual_rate: float = 0.0,
    caveats: tuple[str, ...] = (),
) -> BacktestResult:
    """
    Run `strategy` over `panel` and return the full record of what happened.

    `universe` is required rather than defaulted because it carries the
    disclosures that belong beside every number this returns, and an optional
    argument that carries a disclosure is an omitted disclosure. `caveats` stays
    optional for notes specific to one run.

    `start` only bounds the *trading* window — the strategy still sees history
    before it, which is what makes a twelve-month formation window possible on
    day one instead of a year in.

    `cash_annual_rate` defaults to zero, which understates any strategy that
    holds meaningful cash. That is the safe direction to be wrong in, and Phase 3
    volatility targeting will hold a lot of cash, so it is worth setting
    deliberately rather than leaving at the default and forgetting.
    """
    marks = panel.closes.ffill()
    # Valuation prices for the moment of execution: the day's open where there is
    # one, otherwise the last known close. A holding with no bar today still has a
    # value, and sizing the rest of the book against a NAV that has silently
    # dropped it would sell down every healthy position for no reason.
    open_marks = panel.opens.combine_first(marks)
    adv_notional = (panel.closes * panel.volumes).rolling(ADV_WINDOW).mean().shift(1)
    daily_vol = (np.log(panel.closes / panel.closes.shift(1))
                 .rolling(VOL_WINDOW).std().shift(1))

    dates = _trading_window(panel, start, end)
    if len(dates) < 2:
        raise ValueError("Backtest window needs at least two trading days.")
    rebalance_dates = set(_rebalance_dates(dates, rebalance))

    portfolio = Portfolio(cash=initial_cash)
    pending: pd.Series | None = None
    daily_cash_rate = cash_annual_rate / TRADING_DAYS

    daily_rows, target_history, oversized_fills = [], {}, 0

    for i, date in enumerate(dates):
        portfolio.cash *= 1.0 + daily_cash_rate

        traded_notional, costs_paid = 0.0, 0.0
        if pending is not None:
            open_prices = panel.opens.loc[date]
            deltas = plan_trades(pending, portfolio, open_prices, open_marks.loc[date],
                                 no_trade_band=no_trade_band)
            fills = execute(portfolio, date, deltas, open_prices,
                            adv_notional.loc[date], daily_vol.loc[date], costs)
            traded_notional = sum(f.notional for f in fills)
            costs_paid = sum(f.total_cost for f in fills)
            # Inclusive: a fill whose ADV is unknown is assigned exactly the
            # fallback participation, and an unknown-size trade is the clearest
            # case of a cost estimate that should not pass as a measurement.
            oversized_fills += sum(f.participation >= MAX_CREDIBLE_PARTICIPATION for f in fills)
            pending = None

        close_marks = marks.loc[date]
        nav = portfolio.nav(close_marks)
        daily_rows.append({
            "date": date,
            "nav": nav,
            "cash": portfolio.cash,
            "gross": (nav - portfolio.cash) / nav if nav > 0 else 0.0,
            "n_positions": len(portfolio.shares),
            "traded_notional": traded_notional,
            "costs_paid": costs_paid,
        })

        # No decision on the final day — there is no next open to fill it at.
        if date in rebalance_dates and i < len(dates) - 1:
            pending = target_weights(date, panel, strategy, universe=universe,
                                     min_history=min_history, max_weight=max_weight,
                                     max_gross=max_gross)
            target_history[date] = pending

    daily = pd.DataFrame(daily_rows).set_index("date")
    all_caveats = list(universe.caveats) + list(caveats)
    if oversized_fills:
        all_caveats.append(
            f"{oversized_fills} fills reached {MAX_CREDIBLE_PARTICIPATION:.0%} of average daily volume — "
            "their cost estimates extrapolate the impact model beyond where it was fitted")
    if not target_history:
        # A window shorter than one rebalance period produces a flat curve, no
        # trades and no error. Reporting that as a 0.0% return invites reading a
        # vacuous run as a real one.
        all_caveats.append(
            f"No rebalance occurred: the window holds no {rebalance} period boundary "
            "before its final day, so nothing was ever traded and these figures are empty")
    if daily["cash"].min() < 0:
        all_caveats.append(
            f"Cash went negative (low: {daily['cash'].min():,.0f}) — the run used unpriced "
            "leverage, which no gate here is entitled to reward")

    return BacktestResult(
        equity=daily["nav"],
        daily=daily,
        fills=_fills_frame(portfolio),
        targets=pd.DataFrame(target_history).T.sort_index(),
        metrics=summarize(daily["nav"], daily["costs_paid"], daily["traded_notional"],
                          risk_free_rate=cash_annual_rate),
        meta={
            "strategy": getattr(strategy, "name", type(strategy).__name__),
            "universe_id": universe.id,
            "universe_point_in_time": universe.point_in_time,
            "n_names_first": _tradable_count(panel, dates[0], min_history),
            "n_names_last": _tradable_count(panel, dates[-1], min_history),
            "rebalance": rebalance,
            "no_trade_band": no_trade_band,
            "max_weight": max_weight,
            "max_gross": max_gross,
            "min_history": min_history,
            "initial_cash": initial_cash,
            "cash_annual_rate": cash_annual_rate,
            "costs": costs,
        },
        caveats=all_caveats,
    )


def _tradable_count(panel: PricePanel, date: pd.Timestamp, min_history: int) -> int:
    """How many names were eligible on a date — the cheap tell for a survivor-only universe."""
    return len(panel.as_of(date).tradable_as_of(date, min_history=min_history))


def _trading_window(panel: PricePanel, start, end) -> pd.DatetimeIndex:
    """Dates the backtest may trade on. Rows with no price anywhere are dropped."""
    dates = panel.dates[panel.closes.notna().any(axis=1)]
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return dates


def _rebalance_dates(dates: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    """
    The last trading day of each period — never a calendar date that may be a
    holiday, which is how a rebalance silently goes missing for a month.
    """
    if rebalance not in REBALANCE_RULES:
        raise ValueError(f"rebalance must be one of {sorted(REBALANCE_RULES)}, got {rebalance!r}")
    rule = REBALANCE_RULES[rebalance]
    if rule is None:
        return dates
    return pd.DatetimeIndex(pd.Series(dates, index=dates).resample(rule).last().dropna().values)


# Column order of the fill log. Declared once so the empty frame and the populated
# one cannot describe different schemas — a difference that would only ever show up
# on a run that happened to trade nothing.
FILL_COLUMNS = ["date", "ticker", "shares", "ref_price", "fill_price", "notional",
                "commission", "spread_cost", "impact_cost", "total_cost",
                "participation", "cost_bps"]


def _fills_frame(portfolio: Portfolio) -> pd.DataFrame:
    """Flatten the fill log, keeping the derived columns a post-mortem always needs."""
    if not portfolio.fills:
        return pd.DataFrame(columns=FILL_COLUMNS)

    frame = pd.DataFrame([{
        "date": f.date, "ticker": f.ticker, "shares": f.shares,
        "ref_price": f.ref_price, "fill_price": f.fill_price, "notional": f.notional,
        "commission": f.commission, "spread_cost": f.spread_cost,
        "impact_cost": f.impact_cost, "total_cost": f.total_cost,
        "participation": f.participation,
    } for f in portfolio.fills])
    frame["cost_bps"] = frame["total_cost"] / frame["notional"].replace(0, np.nan) * 1e4
    return frame[FILL_COLUMNS]
