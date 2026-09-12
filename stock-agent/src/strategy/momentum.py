import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import (COV_LOOKBACK_DAYS, EWMA_LAMBDA, MIN_SELECTED_NAMES,
                    MOMENTUM_LOOKBACK_DAYS, MOMENTUM_SKIP_DAYS, MOMENTUM_TOP_FRACTION,
                    TRADING_DAYS, VOL_SCALE_CAP, VOL_TARGET_ANNUAL)
from src.data.panel import PricePanel

_VOL_FLOOR = 1e-6
MIN_COV_NAMES = 2
MIN_COV_ROWS = 20


def formation_return(closes: pd.Series, lookback: int = MOMENTUM_LOOKBACK_DAYS,
                     skip: int = MOMENTUM_SKIP_DAYS) -> float:
    # Measured between two bars that both sit strictly in the past, skipping the most recent
    # `skip` of them. The skip is the factor's whole point: without it the window ends on the
    # short-term reversal that momentum is documented to be damaged by.
    observed = closes.dropna()
    if len(observed) < lookback + skip + 1:
        return np.nan

    end = observed.iloc[-(skip + 1)]
    start = observed.iloc[-(skip + lookback + 1)]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return np.nan
    return end / start - 1.0


class PanelVolForecast:
    # The out-of-sample panel from `build_oos_vol_panel`. Row `d` is a forecast made from
    # information available at `d`, so reading it at rebalance `d` looks ahead of nothing.

    name = "oos-panel"

    def __init__(self, panel: pd.DataFrame):
        self.panel = panel.sort_index()

    def vols(self, as_of, tickers: list[str], history: PricePanel) -> pd.Series:
        available = self.panel.loc[:as_of]
        if available.empty:
            return pd.Series(np.nan, index=tickers, dtype=float)
        return available.iloc[-1].reindex(tickers).astype(float)


class RealizedVolForecast:
    # RiskMetrics EWMA off the panel's own closes. Parameter-free, so it is honestly
    # out-of-sample at every date without a walk-forward fit, which makes it the scaffold
    # the sizing layer can be proven correct against before the model panel is trusted.

    name = "ewma"

    def __init__(self, lam: float = EWMA_LAMBDA, window: int = COV_LOOKBACK_DAYS):
        self.lam = lam
        self.window = window

    def vols(self, as_of, tickers: list[str], history: PricePanel) -> pd.Series:
        closes = history.closes.loc[:as_of, tickers].tail(self.window + 1)
        log_returns = np.log(closes / closes.shift(1))
        variance = (log_returns ** 2).ewm(alpha=1 - self.lam, adjust=False).mean()
        if variance.empty:
            return pd.Series(np.nan, index=tickers, dtype=float)
        annual = np.sqrt((variance.iloc[-1] * TRADING_DAYS).clip(lower=_VOL_FLOOR))
        return annual.reindex(tickers).astype(float)


class _MomentumSelection:
    # Both arms of the Phase 3 gate must rank and select identically, or the comparison
    # measures the selection as well as the sizing.

    def __init__(self, lookback: int = MOMENTUM_LOOKBACK_DAYS,
                 skip: int = MOMENTUM_SKIP_DAYS,
                 top_fraction: float = MOMENTUM_TOP_FRACTION,
                 min_names: int = MIN_SELECTED_NAMES):
        if not 0 < top_fraction <= 1:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")
        if min_names < 1:
            raise ValueError(f"min_names must be at least 1, got {min_names}")
        self.lookback = lookback
        self.skip = skip
        self.top_fraction = top_fraction
        self.min_names = min_names

    def select(self, history: PricePanel, as_of, candidates: list[str]) -> list[str]:
        closes = history.closes.loc[:as_of]
        scores = pd.Series(
            {t: formation_return(closes[t], self.lookback, self.skip)
             for t in candidates if t in closes.columns},
            dtype=float,
        )
        scores = scores[np.isfinite(scores)]
        if scores.empty:
            return []

        # The floor is a risk-limit consequence, not a view: a decile of this universe is
        # eight names, and eight names under a 10% per-name cap cannot be held at full
        # investment, so the cap would quietly convert the remainder into cash.
        n = max(1, self.min_names, int(round(len(scores) * self.top_fraction)))
        return sorted(scores.nlargest(min(n, len(scores))).index)


class MomentumStrategy(_MomentumSelection):
    # The gate's control arm: same names, no sizing. Any difference between this and the
    # vol-targeted run is attributable to the sizing layer alone.

    name = "momentum"
    proposal_scale = "absolute"

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        selected = self.select(history, as_of, candidates)
        if not selected:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(selected), index=selected)


class VolTargetedMomentum(_MomentumSelection):
    name = "vol_targeted_momentum"
    proposal_scale = "absolute"

    def __init__(self, vol_forecast, vol_target: float = VOL_TARGET_ANNUAL,
                 cov_lookback: int = COV_LOOKBACK_DAYS,
                 scale_cap: float = VOL_SCALE_CAP, **selection):
        super().__init__(**selection)
        if not np.isfinite(vol_target) or vol_target <= 0:
            raise ValueError(f"vol_target must be positive and finite, got {vol_target!r}")
        self.vol_forecast = vol_forecast
        self.vol_target = vol_target
        self.cov_lookback = cov_lookback
        self.scale_cap = scale_cap

    def propose(self, as_of: pd.Timestamp, history: PricePanel,
                candidates: list[str]) -> pd.Series:
        selected = self.select(history, as_of, candidates)
        if not selected:
            return pd.Series(dtype=float)

        vols = self.vol_forecast.vols(as_of, selected, history)
        vols = vols[np.isfinite(vols) & (vols > 0)]
        if vols.empty:
            return pd.Series(dtype=float)

        shape = (1.0 / vols) / (1.0 / vols).sum()
        covariance = self._covariance(history, as_of, list(shape.index), vols)
        portfolio_vol = float(np.sqrt(shape.to_numpy() @ covariance @ shape.to_numpy()))
        if not np.isfinite(portfolio_vol) or portfolio_vol <= 0:
            return pd.Series(dtype=float)

        return shape * min(self.vol_target / portfolio_vol, self.scale_cap)

    def _covariance(self, history: PricePanel, as_of, tickers: list[str],
                    vols: pd.Series) -> np.ndarray:
        # Correlations are shrunk, variances come from the forecast. Sample variance would
        # discard the thing Phase 1 was built to supply, and a raw sample correlation on 252
        # days of a concentrated book is the estimate that collapses exactly when it matters.
        sigma = vols[tickers].to_numpy()
        correlation = self._correlation(history, as_of, tickers)
        if correlation is None:
            return np.diag(sigma ** 2)
        return np.diag(sigma) @ correlation @ np.diag(sigma)

    def _correlation(self, history: PricePanel, as_of, tickers: list[str]):
        if len(tickers) < MIN_COV_NAMES:
            return None

        closes = history.closes.loc[:as_of, tickers].tail(self.cov_lookback + 1)
        returns = np.log(closes / closes.shift(1)).dropna()
        if len(returns) < MIN_COV_ROWS:
            return None

        sample = LedoitWolf().fit(returns.to_numpy()).covariance_
        deviations = np.sqrt(np.diag(sample))
        if not np.all(np.isfinite(deviations)) or np.any(deviations <= 0):
            return None

        correlation = sample / np.outer(deviations, deviations)
        np.fill_diagonal(correlation, 1.0)
        return correlation
