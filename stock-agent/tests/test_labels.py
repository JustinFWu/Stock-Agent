# A label reaching one bar too far back hands the model a piece of the present dressed as the
# future, and the resulting score looks like skill. These pin the window at both ends. `label_end`
# gets its own coverage because the walk-forward purge is built on it.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from conftest import make_bars
from config import TRADING_DAYS
from src.labels.target import add_forward_return, add_forward_vol, add_labels

HORIZON = 5


def test_forward_vol_is_the_annualised_vol_of_the_next_h_returns():
    # Computed by hand from the bars the window covers, not from the implementation.
    bars = make_bars(periods=60, seed=21)
    labelled = add_forward_vol(bars, horizon=HORIZON)

    t = 20
    window = bars["Close"].iloc[t:t + HORIZON + 1]          # t .. t+h
    log_returns = np.log(window / window.shift(1)).dropna()  # the h moves after t
    expected = np.sqrt((log_returns ** 2).mean() * TRADING_DAYS)

    assert labelled["forward_vol"].iloc[t] == pytest.approx(expected)


def test_the_label_moves_with_a_bar_inside_its_window():
    # A window that ignores its own contents is not measuring the future.
    bars = make_bars(periods=40, seed=22)
    t = 10

    moved = bars.copy()
    moved.iloc[t + 3, moved.columns.get_loc("Close")] *= 1.20

    before = add_forward_vol(bars, horizon=HORIZON)["forward_vol"].iloc[t]
    after = add_forward_vol(moved, horizon=HORIZON)["forward_vol"].iloc[t]
    assert after != pytest.approx(before)


def test_the_label_ignores_bars_past_the_end_of_its_window():
    # A window that quietly extends past t+h reads further into the future than it advertises, and
    # the saved model's `horizon` field would then describe something the model was never trained
    # on.
    bars = make_bars(periods=40, seed=23)
    t = 10

    moved = bars.copy()
    moved.iloc[t + HORIZON + 1, moved.columns.get_loc("Close")] *= 1.20

    before = add_forward_vol(bars, horizon=HORIZON)["forward_vol"].iloc[t]
    after = add_forward_vol(moved, horizon=HORIZON)["forward_vol"].iloc[t]
    assert after == pytest.approx(before)


def test_label_end_names_the_bar_the_window_closes_on():
    # The purge in the walk-forward split is only as correct as this column.
    bars = make_bars(periods=40, seed=24)
    labelled = add_forward_vol(bars, horizon=HORIZON)

    t = 12
    assert labelled["label_end"].iloc[t] == bars.index[t + HORIZON]


def test_label_end_follows_observed_bars_not_the_calendar():
    # For a ticker with missing sessions the horizon-th *observed* bar is much later than the
    # horizon-th calendar date. That is the case a date-counted embargo gets wrong, and the reason
    # the purge asks the row instead of counting.
    calendar = pd.bdate_range("2021-01-04", periods=20)
    observed = calendar[[0, 1, 2, 3, 4] + list(range(10, 20))]

    bars = make_bars(periods=len(observed), seed=25)
    bars.index = observed
    labelled = add_forward_vol(bars, horizon=HORIZON)

    # The row on the last day before the gap: five observed bars later is day 14,
    # nine calendar days further out than a five-date embargo would assume.
    assert labelled["label_end"].loc[calendar[4]] == calendar[14]


def test_the_last_rows_have_no_label():
    bars = make_bars(periods=30, seed=26)
    labelled = add_forward_vol(bars, horizon=HORIZON)

    assert labelled["forward_vol"].iloc[-HORIZON:].isna().all()
    assert labelled["label_end"].iloc[-HORIZON:].isna().all()


def test_a_flat_window_produces_no_label_rather_than_zero():
    # Zero is fatal downstream: the models train on log(forward_vol), and log(0) is negative
    # infinity. NaN drops the row; zero poisons the fit.
    bars = make_bars(periods=30, seed=27)
    t = 10
    bars.iloc[t:t + HORIZON + 1, bars.columns.get_loc("Close")] = 100.0

    labelled = add_forward_vol(bars, horizon=HORIZON)
    assert pd.isna(labelled["forward_vol"].iloc[t])


def test_forward_return_spans_the_same_horizon():
    bars = make_bars(periods=30, seed=28)
    labelled = add_forward_return(bars, horizon=HORIZON)

    t = 10
    close = bars["Close"]
    expected = (close.iloc[t + HORIZON] - close.iloc[t]) / close.iloc[t]
    assert labelled["future_return"].iloc[t] == pytest.approx(expected)


def test_add_labels_attaches_everything_downstream_requires():
    labelled = add_labels(make_bars(periods=40, seed=29), horizon=HORIZON)
    for column in ("future_return", "forward_vol", "label_end"):
        assert column in labelled.columns
