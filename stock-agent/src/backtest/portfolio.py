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
) -> pd.Series:
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
    """
    nav = portfolio.nav(marks)
    if nav <= 0:
        return pd.Series(dtype=float)

    current = portfolio.weights(marks)
    names = sorted(set(target.index) | set(current.index))

    deltas = {}
    for ticker in names:
        price = prices.get(ticker, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue  # no bar today: halted, delisted, or not yet listed. Cannot trade it.

        target_w = float(target.get(ticker, 0.0))
        current_w = float(current.get(ticker, 0.0))
        drift = target_w - current_w

        is_exit = target_w == 0.0 and current_w != 0.0
        if abs(drift) < no_trade_band and not (is_exit and exit_removed):
            continue

        share_delta = drift * nav / price
        if abs(share_delta) > 1e-9:
            deltas[ticker] = share_delta

    return pd.Series(deltas, dtype=float).sort_index()


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
    and commission, so the one thing the check must not ignore is the costs. The
    estimate is conservative by construction — scaling a trade down reduces its
    market impact, so the realised cost is always at or below the amount reserved.
    """
    sells = share_deltas[share_deltas < 0]
    buys = share_deltas[share_deltas > 0]

    fills = [_book_fill(portfolio, date, t, q, prices, adv_notional, daily_vol, costs)
             for t, q in sells.items()]

    if not buys.empty:
        required = sum(_cash_required(ticker, qty, prices, adv_notional, daily_vol, costs)
                       for ticker, qty in buys.items())
        available = max(portfolio.cash, 0.0)
        scale = min(1.0, available / required) if required > 0 else 0.0
        for ticker, qty in buys.items():
            scaled = qty * scale
            if abs(scaled) > 1e-9:
                fills.append(_book_fill(portfolio, date, ticker, scaled, prices,
                                        adv_notional, daily_vol, costs))

    return fills


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
