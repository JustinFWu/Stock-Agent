"""
The look-ahead firewall.

Every other guarantee in the backtest rests on `as_of` truncation being airtight,
so these tests are blunt about it: after truncation the future bars must not
merely be ignored, they must be absent.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data.universe import UNIVERSE


def test_as_of_removes_future_bars(flat_panel):
    cutoff = flat_panel.dates[100]
    truncated = flat_panel.as_of(cutoff)

    assert truncated.dates.max() == cutoff
    assert len(truncated.dates) == 101  # inclusive of the cutoff day itself
    for frame in (truncated.opens, truncated.highs, truncated.lows,
                  truncated.closes, truncated.volumes):
        assert frame.index.max() == cutoff


def test_as_of_does_not_mutate_the_original(flat_panel):
    original_length = len(flat_panel.dates)
    flat_panel.as_of(flat_panel.dates[10])
    assert len(flat_panel.dates) == original_length


def test_tradable_requires_a_bar_on_the_day(late_lister_panel):
    before_listing = late_lister_panel.dates[299]
    after_listing = late_lister_panel.dates[300]

    assert late_lister_panel.tradable_as_of(before_listing) == ["OLD"]
    assert late_lister_panel.tradable_as_of(after_listing) == ["NEW", "OLD"]


def test_min_history_excludes_a_thin_track_record(late_lister_panel):
    """A name with 50 bars must not be ranked against one with 350."""
    date = late_lister_panel.dates[349]  # NEW has 50 bars here, OLD has 350
    assert late_lister_panel.tradable_as_of(date, min_history=100) == ["OLD"]
    assert late_lister_panel.tradable_as_of(date, min_history=10) == ["NEW", "OLD"]


def test_non_trading_day_is_tradable_by_nobody(flat_panel):
    assert flat_panel.tradable_as_of(pd.Timestamp("1999-01-01")) == []


def test_universe_membership_is_a_no_op_but_honest():
    """
    The membership seam returns everything, and says so via `point_in_time`.

    The flag is what a report branches on. If it ever reads True while
    `members_asof` still returns the full list, the disclosure has become a lie.
    """
    assert UNIVERSE.point_in_time is False
    assert UNIVERSE.members_asof("2008-01-02") == list(UNIVERSE.tickers)
    assert UNIVERSE.members_asof("2026-01-02") == list(UNIVERSE.tickers)
    assert UNIVERSE.caveats, "a non-point-in-time universe must carry a caveat"


def test_universe_spec_is_immutable():
    with pytest.raises(Exception):
        UNIVERSE.point_in_time = True
