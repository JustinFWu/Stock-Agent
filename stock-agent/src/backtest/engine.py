import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (MAX_GROSS, MAX_SECTOR_WEIGHT, MAX_WEIGHT, MIN_HISTORY_DAYS,
                    NO_TRADE_BAND, TRADING_DAYS)
from src.backtest.costs import ALPACA_COSTS, MAX_CREDIBLE_PARTICIPATION, CostModel
from src.backtest.metrics import format_summary, summarize
from src.backtest.portfolio import Portfolio, execute, plan_trades
from src.data.panel import PricePanel
from src.data.universe import UniverseSpec
from src.strategy.weights import Strategy, target_weights

# Built before the strategy on purpose: a backtester written after the signal grows, one
# convenience at a time, into a machine for confirming the signal.

# Each day: fill yesterday's decision at today's OPEN, mark at today's CLOSE, then form
# tomorrow's target if today is a rebalance date. That one day of separation — a signal
# can never be traded at a price used to compute it — is what separates this from fantasy.

# Not modelled, stated plainly because unstated assumptions are how backtests lie: no
# intraday or partial fills, no rejects, no borrow or shorting, no dividends beyond what
# adjusted prices embed, no interest on idle cash unless asked, no taxes. All flatter.

# Two data properties, not loop properties. Bars carry today's adjustment factors, so this
# is not the series a 2006 trader saw; and a name whose bars stop is marked forward forever
# and can never be sold — harmless on survivors, a real problem once delisted names arrive.

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
    # `caveats` is not decoration. Survivorship is the one bias no careful accounting
    # inside the loop can remove, so carrying it on the result means it travels with the
    # number into any report rather than being left in the console.

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
            # A universe that never loses a name across twenty years is survivor-only by
            # construction; printing both counts says so without writing the sentence.
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
    max_sector_weight: float = MAX_SECTOR_WEIGHT,
    cash_annual_rate: float = 0.0,
    caveats: tuple[str, ...] = (),
) -> BacktestResult:
    # `universe` is required rather than defaulted because it carries the disclosures that
    # belong beside every number returned, and an optional argument carrying a disclosure
    # is an omitted disclosure. `caveats` stays optional for notes specific to one run.

    # `start` bounds only the *trading* window — the strategy still sees history before it,
    # which is what makes a twelve-month formation window possible on day one.

    # `cash_annual_rate` defaults to zero, understating any strategy that holds meaningful
    # cash. That is the safe direction, but Phase 3 vol targeting will hold a lot, so it is
    # worth setting deliberately rather than leaving at the default and forgetting.
    _check_run_limits(initial_cash, no_trade_band, max_weight, max_gross)

    marks = panel.closes.ffill()
    # Execution-time valuation: the day's open, else the last close strictly *before*
    # today. A holding with no bar still has a value, and sizing the book against a NAV
    # that silently dropped it would sell down every healthy position for no reason.

    # The one-session shift is the point: today's close is not knowable at today's open, so
    # falling back to it lets a halted name's evening price size this morning's trades in every
    # other name through NAV — measured, that turned one order into a 125-share sale.
    open_marks = panel.opens.combine_first(marks.shift(1))
    adv_notional = (panel.closes * panel.volumes).rolling(ADV_WINDOW).mean().shift(1)
    daily_vol = (np.log(panel.closes / panel.closes.shift(1))
                 .rolling(VOL_WINDOW).std().shift(1))

    dates = _trading_window(panel, start, end)
    if len(dates) < 2:
        raise ValueError("Backtest window needs at least two trading days.")
    rebalance_dates = set(_rebalance_dates(dates, rebalance))

    portfolio = Portfolio(cash=initial_cash)
    pending: pd.Series | None = None
    # Names a previous session meant to trade but found no price for. While set, the pending
    # target is retried for these names only: the rest of the book already reached it, and
    # re-planning would drag every drifting position through the band on an unscheduled day.
    owed: frozenset[str] | None = None
    daily_cash_rate = cash_annual_rate / TRADING_DAYS

    daily_rows, target_history, oversized_fills = [], {}, 0
    deferred_sessions = 0

    for i, date in enumerate(dates):
        portfolio.cash *= 1.0 + daily_cash_rate

        traded_notional, costs_paid = 0.0, 0.0
        if pending is not None:
            open_prices = panel.opens.loc[date]
            plan = plan_trades(pending, portfolio, open_prices, open_marks.loc[date],
                               no_trade_band=no_trade_band, only=owed)
            fills = execute(portfolio, date, plan.deltas, open_prices,
                            adv_notional.loc[date], daily_vol.loc[date], costs)
            traded_notional = sum(f.notional for f in fills)
            costs_paid = sum(f.total_cost for f in fills)
            # Inclusive: a fill whose ADV is unknown is assigned exactly the
            # fallback participation, and an unknown-size trade is the clearest
            # case of a cost estimate that should not pass as a measurement.
            oversized_fills += sum(f.participation >= MAX_CREDIBLE_PARTICIPATION for f in fills)

            # A name with no bar this morning is a trade that has not happened yet,
            # not one that was cancelled. Holding the target open for exactly those
            # names retries it at the next open; the next rebalance supersedes it.
            if plan.blocked:
                owed = frozenset(plan.blocked)
                deferred_sessions += 1
            else:
                pending, owed = None, None

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
                                     max_gross=max_gross,
                                     max_sector_weight=max_sector_weight)
            owed = None
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
    if panel.missing:
        all_caveats.append(
            f"{len(panel.missing)} requested names had no cached bars and were excluded "
            f"({', '.join(panel.missing)}) — this ran on a smaller universe than the one named")
    if deferred_sessions:
        all_caveats.append(
            f"{deferred_sessions} sessions carried an unfilled trade to the next open — "
            "the name had no bar to trade at (a halt, a delisting or a data gap), so the "
            "book sat off its target for those days")
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
            "max_sector_weight": max_sector_weight,
            "min_history": min_history,
            "initial_cash": initial_cash,
            "cash_annual_rate": cash_annual_rate,
            "costs": costs,
        },
        caveats=all_caveats,
    )


def _check_run_limits(initial_cash: float, no_trade_band: float,
                      max_weight: float, max_gross: float) -> None:
    # A NaN limit is the case worth spelling out: every comparison against it is False, so
    # it does not error — it switches the constraint off and returns a plausible equity
    # curve computed without the ceiling anyone thought was applied.
    for name, value in (("initial_cash", initial_cash), ("no_trade_band", no_trade_band),
                        ("max_weight", max_weight), ("max_gross", max_gross)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if initial_cash <= 0:
        raise ValueError(f"initial_cash must be positive, got {initial_cash}")
    if no_trade_band < 0:
        raise ValueError(f"no_trade_band must not be negative, got {no_trade_band}")


def _tradable_count(panel: PricePanel, date: pd.Timestamp, min_history: int) -> int:
    # The cheap tell for a survivor-only universe: this count never falls.
    return len(panel.as_of(date).tradable_as_of(date, min_history=min_history))


def _trading_window(panel: PricePanel, start, end) -> pd.DatetimeIndex:
    dates = panel.dates[panel.closes.notna().any(axis=1)]
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    return dates


def _rebalance_dates(dates: pd.DatetimeIndex, rebalance: str) -> pd.DatetimeIndex:
    # The last trading day of each period, never a calendar date that may be a holiday —
    # which is how a rebalance silently goes missing for a month.
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
