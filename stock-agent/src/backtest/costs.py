"""
Transaction costs.

Most retail backtests die here, and they die quietly: a strategy that rebalances
to an exact volatility target trades constantly, and a cost model of zero turns
that churn into free money. Three components, each modelled separately so the
result can be attributed rather than just totalled.

  commission    Explicit fees. Alpaca's US equity commission is zero, so the
                default is zero — but it stays a parameter because IBKR's is not,
                and the whole point of the broker interface in Phase 4 is that the
                venue can change.

  half-spread   The cost of crossing. A marketable order pays roughly half the
                quoted spread against the midpoint, each way. For S&P 100-scale
                names this is a basis point or two; the default of 2bp is
                deliberately a little pessimistic, since the panel's prices are
                consolidated closes and opens rather than executable quotes.

  impact        The price you move by trading. Modelled with the square-root law,
                impact ~= coefficient * daily_vol * sqrt(shares / ADV), which is
                the standard empirical form (Almgren, Torre, Grinold-Kahn) and the
                only one of the three that scales non-linearly with size. It is
                what stops the backtest from pretending a $50m order fills like a
                $50k one.

Every rate here is an estimate, not a measurement. The honest use of this module
is to check that a strategy survives plausible costs with room to spare — not to
report a net Sharpe to two decimal places and believe it. When Phase 4 produces
real fills, the fitted slippage from those fills replaces these numbers.
"""

import math
from dataclasses import dataclass

# Trading is assumed to be a small share of a name's daily volume. Beyond this the
# square-root law is being extrapolated well past where it was estimated, so the
# engine warns rather than silently reporting a fill it does not believe.
MAX_CREDIBLE_PARTICIPATION = 0.10


@dataclass(frozen=True)
class CostModel:
    """
    Per-trade cost estimate in dollars, split by component.

    Rates are per side: a round trip pays each of these twice.
    """

    commission_bps: float = 0.0
    commission_per_share: float = 0.0
    min_commission: float = 0.0
    half_spread_bps: float = 2.0

    # Coefficient on the square-root law. Published estimates of the full temporary
    # impact cluster near 1.0; a passive-ish execution spread over a day realizes
    # roughly half of it, which is what 0.5 encodes.
    impact_coef: float = 0.5

    # Used when a name's average dollar volume is unknown or zero, so a missing
    # volume figure produces a pessimistic cost rather than a free trade.
    fallback_participation: float = MAX_CREDIBLE_PARTICIPATION

    # The same principle for the other input to impact. Without it the volatility
    # leg silently defeats the participation fallback: ADV and daily volatility are
    # computed from the same bars over the same rolling window, so whenever one is
    # missing the other is too, and a missing volatility used to short-circuit
    # impact to zero — making the pessimistic ADV fallback unreachable in practice.
    # 3% a day, roughly 48% annualized, is deliberately high.
    fallback_daily_vol: float = 0.03

    def commission(self, shares: float, notional: float) -> float:
        """Explicit fees on one trade. `shares` and `notional` are absolute values."""
        fee = notional * self.commission_bps / 1e4 + shares * self.commission_per_share
        return max(fee, self.min_commission) if notional > 0 else 0.0

    def slippage_rate(self, notional: float, adv_notional: float, daily_vol: float) -> float:
        """
        Fraction of notional lost to spread and impact on one trade.

        Returned as a rate rather than dollars so the engine can apply it to the
        fill price, which is where it actually shows up: you do not pay slippage
        out of a separate account, you pay it by transacting at a worse price.
        """
        return self.half_spread_bps / 1e4 + self._impact_rate(notional, adv_notional, daily_vol)

    def participation(self, notional: float, adv_notional: float) -> float:
        """Trade size as a fraction of the name's average daily dollar volume."""
        if adv_notional is None or not math.isfinite(adv_notional) or adv_notional <= 0:
            return self.fallback_participation
        return abs(notional) / adv_notional

    def _impact_rate(self, notional: float, adv_notional: float, daily_vol: float) -> float:
        # An *unknown* volatility and a *measured* zero are different facts. The
        # first is missing data and gets the pessimistic fallback; the second says
        # the name genuinely did not move, and impact is a volatility-scaled
        # quantity, so there is nothing to move.
        if daily_vol is None or not math.isfinite(daily_vol):
            daily_vol = self.fallback_daily_vol
        if daily_vol <= 0:
            return 0.0
        return self.impact_coef * daily_vol * math.sqrt(self.participation(notional, adv_notional))


# Alpaca paper trading, the Phase 4 venue: no commission, and the spread and impact
# assumptions above. This is the default the backtest runs with.
ALPACA_COSTS = CostModel()

# A deliberately harsh model. If a strategy's edge survives this it is not a
# costing artefact; if it only works under ALPACA_COSTS, the margin is thinner
# than the uncertainty in the cost estimate itself.
PESSIMISTIC_COSTS = CostModel(half_spread_bps=5.0, impact_coef=1.0)

# Costs switched off. Only for measuring the gross-to-net gap — never a result.
ZERO_COSTS = CostModel(half_spread_bps=0.0, impact_coef=0.0, fallback_participation=0.0,
                       fallback_daily_vol=0.0)
