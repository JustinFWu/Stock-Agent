import math
from dataclasses import dataclass

# Three components kept separate so a result can be attributed, not just totalled. Every
# rate here is an estimate: the honest use is checking a strategy survives plausible costs
# with room to spare. Phase 4's real fills replace these numbers with fitted slippage.

# Trading is assumed to be a small share of a name's daily volume. Beyond this the
# square-root law is being extrapolated well past where it was estimated, so the
# engine warns rather than silently reporting a fill it does not believe.
MAX_CREDIBLE_PARTICIPATION = 0.10


@dataclass(frozen=True)
class CostModel:
    # Rates are per side: a round trip pays each of these twice.

    # Alpaca's US equity commission is zero, but these stay parameters because IBKR's is
    # not, and the point of the Phase 4 broker interface is that the venue can change.
    commission_bps: float = 0.0
    commission_per_share: float = 0.0
    min_commission: float = 0.0

    # A marketable order pays roughly half the quoted spread each way. 2bp is deliberately
    # pessimistic for S&P 100-scale names: the panel holds consolidated closes and opens,
    # not executable quotes.
    half_spread_bps: float = 2.0

    # Coefficient on the square-root law. Published estimates of the full temporary
    # impact cluster near 1.0; a passive-ish execution spread over a day realizes
    # roughly half of it, which is what 0.5 encodes.
    impact_coef: float = 0.5

    # Used when a name's average dollar volume is unknown or zero, so a missing
    # volume figure produces a pessimistic cost rather than a free trade.
    fallback_participation: float = MAX_CREDIBLE_PARTICIPATION

    # ADV and daily vol come from the same bars over the same window, so whenever one is
    # missing the other is too. Without this fallback a missing vol short-circuits impact
    # to zero and the pessimistic ADV fallback becomes unreachable. 3%/day is high on purpose.
    fallback_daily_vol: float = 0.03

    def __post_init__(self) -> None:
        # A negative rate does not understate costs, it inverts them: trading becomes a
        # source of return and the backtest rewards churn. Zero stays legal — ZERO_COSTS
        # is a deliberate measurement of the gross-to-net gap.
        for name in ("commission_bps", "commission_per_share", "min_commission",
                     "half_spread_bps", "impact_coef", "fallback_participation",
                     "fallback_daily_vol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")

    def commission(self, shares: float, notional: float) -> float:
        # `shares` and `notional` are absolute values.
        fee = notional * self.commission_bps / 1e4 + shares * self.commission_per_share
        return max(fee, self.min_commission) if notional > 0 else 0.0

    def slippage_rate(self, notional: float, adv_notional: float, daily_vol: float) -> float:
        # A rate rather than dollars so the engine applies it to the fill price, which is
        # where it shows up: you pay slippage by transacting worse, not from a side account.
        return self.half_spread_bps / 1e4 + self._impact_rate(notional, adv_notional, daily_vol)

    def participation(self, notional: float, adv_notional: float) -> float:
        if adv_notional is None or not math.isfinite(adv_notional) or adv_notional <= 0:
            return self.fallback_participation
        return abs(notional) / adv_notional

    def _impact_rate(self, notional: float, adv_notional: float, daily_vol: float) -> float:
        # An unknown volatility and a measured zero are different facts: the first is
        # missing data and gets the fallback, the second says the name did not move, and
        # impact is volatility-scaled, so there is nothing to move.
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
