"""
Portfolio accounting and order generation.

Deliberately dumb about strategy and deliberately careful about cash: every
dollar is either in a position or in the cash balance, and the only way shares
change is through a fill that also moves cash. If those two ever disagree the
equity curve is fiction, so `apply_fill` is the single mutation point.

The no-trade band lives here too, because it is an accounting decision rather
than a strategy one. Volatility targeting produces a slightly different ideal
weight every single day; following it exactly means trading every day, and the
turnover eats more than the tracking error it removes. The band says: only move
when the portfolio has drifted far enough from target to be worth paying for.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.backtest.costs import CostModel


@dataclass(frozen=True)
class Fill:
    """One executed trade, with its costs itemised for attribution."""

    date: pd.Timestamp
    ticker: str
    shares: float           # signed: positive buys, negative sells
    ref_price: float        # the untouched market price the fill was measured against
    fill_price: float       # what was actually paid, after spread and impact
    commission: float
    spread_cost: float
    impact_cost: float
    participation: float    # trade size as a fraction of average daily dollar volume

    @property
    def notional(self) -> float:
        return abs(self.shares) * self.ref_price

    @property
    def total_cost(self) -> float:
        return self.commission + self.spread_cost + self.impact_cost


# A drift worth less than this in dollars is not a trade. Zero drift passes the
# band check when the band is itself zero, and calling that a trade would defer a
# name with nothing owed on it forever.
NEGLIGIBLE_NOTIONAL = 1e-9

# Bisection steps used to size buys against available cash. 2^-40 of an order is
# well below a cent on any realistic book, and the loop is only entered when the
# full order does not fit.
AFFORDABILITY_ITERATIONS = 40


@dataclass(frozen=True)
class TradePlan:
    """
    What to trade, and what could not be traded.

    `blocked` is the part a bare Series of deltas cannot express: names the plan
    genuinely wanted to move but found no execution price for. Without it the
    caller cannot tell "nothing to do here" apart from "this trade did not
    happen", and those call for opposite responses.
    """

    deltas: pd.Series
    blocked: tuple[str, ...] = ()


@dataclass
class Portfolio:
    """Cash plus share counts. Fractional shares are allowed — Alpaca supports them."""

    cash: float
    shares: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)

    def position_value(self, prices: pd.Series) -> float:
        """
        Marked value of all holdings.

        The engine passes forward-filled closes for marking, so a name that is
        merely missing a bar holds its last price rather than dropping to zero and
        printing a round trip of fake loss and fake gain in the equity curve. The
        zero fallback below is for a name that has never had a price at all, which
        can only be a bug upstream.
        """
        return float(sum(qty * _price_or_zero(prices, ticker) for ticker, qty in self.shares.items()))

    def nav(self, prices: pd.Series) -> float:
        return self.cash + self.position_value(prices)

    def weights(self, prices: pd.Series) -> pd.Series:
        """Current portfolio weights. Sums to less than 1 by the cash fraction."""
        nav = self.nav(prices)
        if nav <= 0:
            return pd.Series(dtype=float)
        held = {t: q * _price_or_zero(prices, t) / nav for t, q in self.shares.items() if q != 0}
        return pd.Series(held, dtype=float).sort_index()

    def apply_fill(self, fill: Fill) -> None:
        """
        The only place shares and cash change.

        Slippage is already inside `fill_price`, so the cash movement is just
        signed notional at the fill price plus the explicit commission. Costs are
        never subtracted twice — the itemised spread and impact figures on the
        fill exist for reporting, not for accounting.
        """
        self.cash -= fill.shares * fill.fill_price + fill.commission
        self.shares[fill.ticker] = self.shares.get(fill.ticker, 0.0) + fill.shares
        if abs(self.shares[fill.ticker]) < 1e-9:
            self.shares.pop(fill.ticker)
        self.fills.append(fill)


def plan_trades(
    target: pd.Series,
    portfolio: Portfolio,
    prices: pd.Series,
    marks: pd.Series,
    *,
    no_trade_band: float,
    exit_removed: bool = True,
    only: frozenset[str] | None = None,
) -> TradePlan:
    """
    Share deltas that move the portfolio toward `target`, subject to the band.

    Two price vectors, and the distinction is load-bearing. `prices` are the raw
    bars a trade would actually execute at — a name with no bar cannot be traded
    at any price. `marks` are the valuation prices, carried forward when a bar is
    missing, and they are what NAV and the current weights are computed from.

    Sizing against raw prices instead would value a halted holding at zero, drop
    NAV by that name's full weight, and then dutifully sell down every *healthy*
    position to hit its target share of the smaller portfolio — a fake loss, a
    fake recovery the next day when the bar returns, and real costs on a trade
    that should never have been placed.

    Names drifting inside the band are left alone. A name the strategy has
    dropped entirely is exited regardless of the band when `exit_removed` is set:
    a position the strategy no longer wants is a risk decision, not a rebalancing
    nicety, and leaving 30bp of a name the signal rejected accumulates into a
    portfolio nobody chose.

    Trades that clear the band go the full distance to target rather than only to
    the band edge. Trading to the edge would leave the portfolio permanently at
    its maximum tolerated error and guarantee another trade shortly after.

    A name the plan wants to move but has no price for is reported in `blocked`
    rather than quietly omitted. It is the difference between a trade nobody asked
    for and a trade that could not be placed, and only the caller knows whether to
    retry it — see the engine, which carries blocked names to the next session.

    `only` restricts the plan to a subset of names, which is how that retry stays
    surgical. Re-planning the whole target would drag every position that has
    since drifted back through the band on a day the strategy never asked to
    rebalance, turning one halted name into a portfolio-wide trade.
    """
    nav = portfolio.nav(marks)
    if nav <= 0:
        return TradePlan(deltas=pd.Series(dtype=float))

    current = portfolio.weights(marks)
    names = sorted(set(target.index) | set(current.index))
    if only is not None:
        names = [t for t in names if t in only]

    deltas, blocked = {}, []
    for ticker in names:
        target_w = float(target.get(ticker, 0.0))
        current_w = float(current.get(ticker, 0.0))
        drift = target_w - current_w

        is_exit = target_w == 0.0 and current_w != 0.0
        wants_trade = abs(drift) >= no_trade_band or (is_exit and exit_removed)
        if not wants_trade or abs(drift) * nav <= NEGLIGIBLE_NOTIONAL:
            continue

        # Tradability is checked *after* the decision to trade, not before. Checked
        # first, a halted name lands in the same branch as a name nobody wanted to
        # touch, and the information that a wanted trade did not happen is gone.
        price = prices.get(ticker, np.nan)
        if not np.isfinite(price) or price <= 0:
            blocked.append(ticker)  # halted, delisted, or not yet listed
            continue

        share_delta = drift * nav / price
        if abs(share_delta) > 1e-9:
            deltas[ticker] = share_delta

    return TradePlan(deltas=pd.Series(deltas, dtype=float).sort_index(),
                     blocked=tuple(blocked))


def execute(
    portfolio: Portfolio,
    date: pd.Timestamp,
    share_deltas: pd.Series,
    prices: pd.Series,
    adv_notional: pd.Series,
    daily_vol: pd.Series,
    costs: CostModel,
) -> list[Fill]:
    """
    Turn planned share deltas into fills, applying costs and the cash constraint.

    Sells settle before buys, which is both how the cash actually becomes
    available and what keeps a fully-invested rebalance from needing a margin
    loan it never asked for. If buying power still falls short the buys are scaled
    down proportionally rather than partially dropped, so the portfolio stays close
    to the intended shape instead of silently over-weighting whichever names
    sorted first.

    The scaling is measured against the *cost-inclusive* price of the buys, not
    the raw notional. Measuring it against raw notional is how a long-only
    backtest quietly ends up on margin: the shortfall is precisely the slippage
    and commission, so the one thing the check must not ignore is the costs.
    `_affordable_scale` works out how far the buys have to shrink, without
    assuming the costs shrink with them.
    """
    sells = share_deltas[share_deltas < 0]
    buys = share_deltas[share_deltas > 0]

    fills = [_book_fill(portfolio, date, t, q, prices, adv_notional, daily_vol, costs)
             for t, q in sells.items()]

    if not buys.empty:
        scale = _affordable_scale(buys, prices, adv_notional, daily_vol, costs,
                                  available=max(portfolio.cash, 0.0))
        for ticker, qty in buys.items():
            scaled = qty * scale
            if abs(scaled) > 1e-9:
                fills.append(_book_fill(portfolio, date, ticker, scaled, prices,
                                        adv_notional, daily_vol, costs))

    return fills


def _affordable_scale(buys: pd.Series, prices: pd.Series, adv_notional: pd.Series,
                      daily_vol: pd.Series, costs: CostModel, *, available: float) -> float:
    """
    The largest fraction of the planned buys that the cash on hand actually covers.

    Dividing available cash by required cash is only correct when every component
    of the cost scales with the trade, and a *minimum* commission does not: shrink
    the order and the fee stays exactly where it was. The naive scale therefore
    reserves less than the smaller order goes on to cost, and the account finishes
    overdrawn — with 100 in cash, two shares at 100 and a 1.00 minimum fee, it
    lands at -0.50. Impact fails the same assumption in the opposite, harmless
    direction, growing as the square root of size rather than linearly.

    Cash required is monotone non-decreasing in the scale whatever the cost model
    does in between, so a bisection finds the largest affordable fraction without
    needing to invert it. Forty halvings resolve the fraction far below a cent on
    any book this engine will see. A scale of zero — the honest answer when the
    minimum fees alone exceed the balance — simply places no buys.
    """
    def required(scale: float) -> float:
        return sum(_cash_required(ticker, qty * scale, prices, adv_notional, daily_vol, costs)
                   for ticker, qty in buys.items())

    if required(1.0) <= available:
        return 1.0

    low, high = 0.0, 1.0
    for _ in range(AFFORDABILITY_ITERATIONS):
        mid = (low + high) / 2.0
        if required(mid) <= available:
            low = mid
        else:
            high = mid
    return low


def _cash_required(ticker: str, shares: float, prices: pd.Series, adv_notional: pd.Series,
                   daily_vol: pd.Series, costs: CostModel) -> float:
    """Cash a buy will actually consume: notional at the fill price, plus commission."""
    ref_price = float(prices[ticker])
    notional = abs(shares) * ref_price
    slippage = costs.slippage_rate(notional, float(adv_notional.get(ticker, np.nan)),
                                   float(daily_vol.get(ticker, np.nan)))
    return notional * (1.0 + slippage) + costs.commission(abs(shares), notional)


def _book_fill(portfolio: Portfolio, date: pd.Timestamp, ticker: str, shares: float,
               prices: pd.Series, adv_notional: pd.Series, daily_vol: pd.Series,
               costs: CostModel) -> Fill:
    """Price one trade through the cost model and book it into the portfolio."""
    ref_price = float(prices[ticker])
    notional = abs(shares) * ref_price
    adv = float(adv_notional.get(ticker, np.nan))
    vol = float(daily_vol.get(ticker, np.nan))

    slippage = costs.slippage_rate(notional, adv, vol)
    side = 1.0 if shares > 0 else -1.0
    fill_price = ref_price * (1.0 + side * slippage)

    spread_cost = notional * costs.half_spread_bps / 1e4
    fill = Fill(
        date=date,
        ticker=ticker,
        shares=shares,
        ref_price=ref_price,
        fill_price=fill_price,
        commission=costs.commission(abs(shares), notional),
        spread_cost=spread_cost,
        impact_cost=notional * slippage - spread_cost,
        participation=costs.participation(notional, adv),
    )
    portfolio.apply_fill(fill)
    return fill


def _price_or_zero(prices: pd.Series, ticker: str) -> float:
    price = prices.get(ticker, np.nan)
    return float(price) if np.isfinite(price) else 0.0
