from stock_agent.config import VOL_TARGET_ANNUAL
from stock_agent.strategy.momentum import (
    MomentumStrategy,
    PanelVolForecast,
    RealizedVolForecast,
    VolTargetedMomentum,
)
from stock_agent.strategy.weights import EqualWeightStrategy, Strategy

# One place that turns a strategy name into a strategy, because two callers now need it:
# the research CLI, which backtests it, and the session CLI, which trades it. Those two
# asking different factories for "vol-momentum" is the same class of divergence the single
# weight path exists to prevent, one level up.

STRATEGIES = ("equal", "momentum", "vol-momentum")
VOL_SOURCES = ("panel", "ewma")


def build_strategy(name: str, vol_source: str = "panel",
                   vol_target: float = VOL_TARGET_ANNUAL) -> Strategy:
    if name == "equal":
        return EqualWeightStrategy()
    if name == "momentum":
        return MomentumStrategy()
    if name != "vol-momentum":
        raise ValueError(f"unknown strategy {name!r}; expected one of {list(STRATEGIES)}")

    # Imported where it is used rather than at module scope: loading the walk-forward panel
    # pulls in the model stack and reads a saved artefact, and the equal-weight and plain
    # momentum paths have no business paying for either.
    if vol_source == "ewma":
        return VolTargetedMomentum(RealizedVolForecast(), vol_target=vol_target)

    from stock_agent.models.vol_forecast import load_oos_vol_panel
    return VolTargetedMomentum(PanelVolForecast(load_oos_vol_panel()), vol_target=vol_target)
