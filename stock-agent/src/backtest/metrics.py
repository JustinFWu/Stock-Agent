"""
Performance statistics for an equity curve.

Kept separate from the engine so the same numbers can be computed for a live
track record later without a backtest anywhere in the call stack — the Phase 4
monitoring surface reports against the same definitions the gates were written
in, which is only true if there is one definition of each.

Two things reported here that a naive summary leaves out, both because they are
where a plausible-looking strategy usually fails:

  turnover    Annualised two-way turnover. A Sharpe that needs 900% turnover a
              year is a Sharpe made of cost assumptions.
  cost drag   Realised costs as an annualised fraction of average NAV, so the
              gross-to-net gap is a stated number rather than an inference.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TRADING_DAYS

# Below this daily standard deviation a return series is flat, and the float noise
# left in it is not risk. Testing `std > 0` is not enough: a portfolio sitting in
# cash at a fixed rate has a std around 1e-18, which divides into a Sharpe of
# 5e13. Volatility targeting will put the portfolio entirely in cash sooner or
# later, so this is a case the reporting has to survive rather than an edge case.
FLAT_RETURN_TOLERANCE = 1e-12


def summarize(equity: pd.Series, costs_paid: pd.Series, traded_notional: pd.Series,
              risk_free_rate: float = 0.0) -> dict:
    """
    Headline statistics for a daily NAV series.

    `costs_paid` and `traded_notional` are daily dollar totals aligned to the same
    index. Sharpe is computed on daily simple returns, in excess of a constant
    risk-free rate, and annualised by sqrt(252) — the usual convention, and one
    that overstates the ratio when returns are autocorrelated or fat-tailed.
    Treat a Sharpe near a gate threshold as inconclusive, not as a pass.
    """
    equity = equity.dropna()
    if len(equity) < 2:
        raise ValueError("Need at least two NAV observations to summarize.")

    returns = equity.pct_change().dropna()
    # N daily marks span N-1 periods of growth, not N. Counting the marks instead
    # of the periods stretches the elapsed time by a day and quietly shaves the
    # CAGR, the turnover and the cost drag — small, but wrong in the direction
    # that flatters, and it compounds into every gate this feeds.
    years = len(returns) / TRADING_DAYS
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    sd = float(returns.std(ddof=1))
    ann_vol = sd * np.sqrt(TRADING_DAYS)
    excess = returns - risk_free_rate / TRADING_DAYS
    sharpe = (float(excess.mean() / sd * np.sqrt(TRADING_DAYS))
              if sd > FLAT_RETURN_TOLERANCE else np.nan)

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else np.nan

    # Both ratios are accumulated day by day against that day's NAV, not against
    # the average NAV over the whole run. On a curve that compounds 18x, dividing
    # early trading by a mean NAV several times its actual size understates
    # turnover and cost drag by nearly 20% — and understates them in the
    # flattering direction, in the two statistics that exist precisely because
    # they are where a plausible-looking strategy fails.
    total_costs = float(costs_paid.sum())
    total_traded = float(traded_notional.sum())
    turnover_ratio = _sum_over_nav(traded_notional, equity)
    cost_ratio = _sum_over_nav(costs_paid, equity)

    return {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "years": round(years, 2),
        "total_return": float(total_return),
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else np.nan,
        "ann_turnover": float(turnover_ratio / years) if years > 0 else np.nan,
        "total_costs": total_costs,
        "total_traded": total_traded,
        "cost_drag_ann": float(cost_ratio / years) if years > 0 else np.nan,
        # Days on which anything traded, not a count of trades. Named for what it
        # measures so a consumer reading the dict cannot mistake it for a fill count.
        "n_trade_days": int((traded_notional > 0).sum()),
    }


def _sum_over_nav(flows: pd.Series, equity: pd.Series) -> float:
    """Sum each day's dollar flow as a fraction of that day's NAV."""
    aligned = pd.concat([flows.rename("flow"), equity.rename("nav")], axis=1).dropna()
    aligned = aligned[aligned["nav"] > 0]
    return float((aligned["flow"] / aligned["nav"]).sum())


def summarize_relative(strategy_equity: pd.Series, baseline_equity: pd.Series) -> dict:
    """
    Statistics of the strategy's returns minus a baseline run on the same universe.

    This exists because the absolute Sharpe of anything run on a survivorship-
    biased universe is uninterpretable: equal-weight buy-and-hold of this
    project's 82 names scores about 0.91 with no signal in it at all. Differencing
    against a baseline that inherits the identical bias cancels most of it, which
    makes the information ratio below the only figure here that is about the
    strategy rather than about the universe.

    Both series are reported alongside the difference rather than collapsed into
    one adjusted number. A haircut destroys the audit trail — once one figure has
    been adjusted, a reader cannot tell which of the others are measurements.
    """
    aligned = pd.concat([strategy_equity.rename("strategy"),
                         baseline_equity.rename("baseline")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("Strategy and baseline curves do not overlap on enough dates.")

    strat_ret = aligned["strategy"].pct_change().dropna()
    base_ret = aligned["baseline"].pct_change().dropna()
    active = strat_ret - base_ret

    years = len(active) / TRADING_DAYS  # return periods, not marks — see `summarize`
    tracking_error = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))

    def _cagr(curve: pd.Series) -> float:
        return float((curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1.0) if years > 0 else np.nan

    def _sharpe(returns: pd.Series) -> float:
        sd = float(returns.std(ddof=1))
        return (float(returns.mean() / sd * np.sqrt(TRADING_DAYS))
                if sd > FLAT_RETURN_TOLERANCE else np.nan)

    base_var = float(base_ret.var(ddof=1))
    return {
        "sharpe_strategy": _sharpe(strat_ret),
        "sharpe_baseline": _sharpe(base_ret),
        "sharpe_diff": _sharpe(strat_ret) - _sharpe(base_ret),
        "excess_cagr": _cagr(aligned["strategy"]) - _cagr(aligned["baseline"]),
        "tracking_error": tracking_error,
        # A strategy that is its own baseline has no active risk and therefore no
        # information ratio. NaN is the honest answer; a huge number would not be.
        "information_ratio": (float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))
                              if float(active.std(ddof=1)) > FLAT_RETURN_TOLERANCE else np.nan),
        "beta_to_baseline": (float(strat_ret.cov(base_ret) / base_var) if base_var > 0 else np.nan),
    }


def format_relative_summary(relative: dict, baseline_name: str) -> str:
    """The relative block, with the information ratio last because it is the conclusion."""
    return "\n".join([
        f"  vs baseline     {baseline_name}",
        f"  Sharpe          {relative['sharpe_strategy']:>8.2f}  strategy",
        f"                  {relative['sharpe_baseline']:>8.2f}  baseline",
        f"  excess CAGR     {relative['excess_cagr']:>8.2%}",
        f"  tracking error  {relative['tracking_error']:>8.2%}",
        f"  beta            {relative['beta_to_baseline']:>8.2f}",
        f"  info ratio      {relative['information_ratio']:>8.2f}   <- the bias-cancelling number",
    ])


def format_summary(metrics: dict) -> str:
    """Human-readable block, in the order a reader actually wants to check it."""
    lines = [
        f"  period          {metrics['start']} -> {metrics['end']}  ({metrics['years']:.1f}y)",
        f"  total return    {metrics['total_return']:>8.1%}",
        f"  CAGR            {metrics['cagr']:>8.2%}",
        f"  ann vol         {metrics['ann_vol']:>8.2%}",
        f"  Sharpe          {metrics['sharpe']:>8.2f}",
        f"  max drawdown    {metrics['max_drawdown']:>8.1%}",
        f"  Calmar          {metrics['calmar']:>8.2f}",
        f"  ann turnover    {metrics['ann_turnover']:>8.0%}",
        f"  cost drag /yr   {metrics['cost_drag_ann']:>8.2%}",
        f"  costs paid      {metrics['total_costs']:>8,.0f}   over {metrics['n_trade_days']} trading days",
    ]
    return "\n".join(lines)
