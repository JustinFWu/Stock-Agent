# The control the roadmap's architecture names and the repo did not have. The property
# worth testing hardest is not that it trips — it is that the trip survives the restart,
# because the restart is very often caused by whatever tripped it.

import json

import pytest

from stock_agent.execution.killswitch import KillSwitch, TripReason, evaluate_tripwires


def switch(tmp_path, **kwargs) -> KillSwitch:
    return KillSwitch(tmp_path / "kill_switch.json", **kwargs)


def test_a_fresh_switch_is_armed(tmp_path):
    assert not switch(tmp_path).is_tripped


def test_a_drawdown_past_the_limit_trips_it(tmp_path):
    breaker = switch(tmp_path, max_drawdown=0.20)
    breaker.check(equity=100_000.0, positions={})
    breaches = breaker.check(equity=75_000.0, positions={})

    assert breaker.is_tripped
    assert [b.reason for b in breaches] == [TripReason.DRAWDOWN]


def test_a_drawdown_inside_the_limit_does_not(tmp_path):
    breaker = switch(tmp_path, max_drawdown=0.20)
    breaker.check(equity=100_000.0, positions={})
    breaker.check(equity=85_000.0, positions={})

    assert not breaker.is_tripped


def test_the_peak_is_read_before_this_observation_is_folded_in(tmp_path):
    # Updating the high-water mark first would make every observation its own peak and
    # measure every drawdown as zero. The first call must not trip on itself.
    breaker = switch(tmp_path, max_drawdown=0.20)
    assert breaker.check(equity=50_000.0, positions={}) == ()
    assert breaker.peak_equity == 50_000.0


def test_the_peak_only_ever_rises(tmp_path):
    breaker = switch(tmp_path, max_drawdown=0.50)
    breaker.check(equity=100_000.0, positions={})
    breaker.check(equity=90_000.0, positions={})

    assert breaker.peak_equity == 100_000.0


def test_a_trip_survives_a_restart(tmp_path):
    # The whole reason the state is on disk. A switch held in memory clears itself on the
    # restart that whatever tripped it just caused.
    breaker = switch(tmp_path, max_drawdown=0.20)
    breaker.check(equity=100_000.0, positions={})
    breaker.check(equity=50_000.0, positions={})

    restarted = switch(tmp_path)
    assert restarted.is_tripped
    assert restarted.state.reason is TripReason.DRAWDOWN
    assert restarted.peak_equity == 100_000.0


def test_the_peak_survives_a_restart_too(tmp_path):
    switch(tmp_path).check(equity=250_000.0, positions={})
    assert switch(tmp_path).peak_equity == 250_000.0


def test_the_first_trip_wins(tmp_path):
    # The first cause is the diagnosis; whatever fires next is usually its consequence.
    breaker = switch(tmp_path)
    breaker.trip(TripReason.DRAWDOWN, "first")
    breaker.trip(TripReason.NAV_MISMATCH, "second")

    assert breaker.state.reason is TripReason.DRAWDOWN
    assert breaker.state.detail == "first"


def test_a_reset_needs_a_name_on_it(tmp_path):
    breaker = switch(tmp_path)
    breaker.trip(TripReason.DRAWDOWN, "measured")

    with pytest.raises(ValueError, match="whoever is clearing it"):
        breaker.reset("")
    with pytest.raises(ValueError, match="whoever is clearing it"):
        breaker.reset("   ")
    assert breaker.is_tripped


def test_a_manual_reset_clears_it_and_is_recorded(tmp_path):
    breaker = switch(tmp_path)
    breaker.check(equity=100_000.0, positions={})
    breaker.trip(TripReason.DRAWDOWN, "measured")
    breaker.reset("operator", note="looked at it")

    reopened = switch(tmp_path)
    assert not reopened.is_tripped
    assert reopened.state.cleared_by == "operator"
    # Clearing the switch must not clear the high-water mark, or the next session measures
    # its drawdown from wherever the book happens to sit after the incident.
    assert reopened.peak_equity == 100_000.0


def test_an_unreadable_state_file_refuses_to_start(tmp_path):
    # "Assume armed" is the tempting default and the wrong one: a kill switch whose state
    # cannot be read is a kill switch whose state is unknown.
    path = tmp_path / "kill_switch.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="whether trading is halted is unknown"):
        KillSwitch(path)


def test_the_state_file_is_plain_readable_json(tmp_path):
    # Read by a human at 3am deciding whether to clear it.
    breaker = switch(tmp_path)
    breaker.trip(TripReason.THIN_COVERAGE, "only 40%")

    written = json.loads((tmp_path / "kill_switch.json").read_text(encoding="utf-8"))
    assert written["tripped"] is True
    assert written["reason"] == "thin_coverage"


def test_a_short_position_trips_it(tmp_path):
    breaker = switch(tmp_path)
    breaches = breaker.check(equity=100_000.0, positions={"AAA": 10.0, "BBB": -5.0})

    assert breaker.is_tripped
    assert breaches[0].reason is TripReason.UNEXPECTED_POSITION
    assert "BBB" in breaches[0].detail


def test_a_holding_we_cannot_price_trips_it(tmp_path):
    breaker = switch(tmp_path)
    breaches = breaker.check(equity=100_000.0, positions={"AAA": 10.0},
                             unpriced_positions=("ZZZ",))

    assert breaches[0].reason is TripReason.UNEXPECTED_POSITION
    assert breaker.is_tripped


def test_every_breach_is_reported_not_just_the_first(tmp_path):
    # A drawdown arriving alongside an unexpected position is a different morning from a
    # drawdown on its own, and the alert has to be able to say so.
    breaker = switch(tmp_path, max_drawdown=0.10)
    breaker.check(equity=100_000.0, positions={})
    breaches = breaker.check(equity=50_000.0, positions={"BBB": -1.0}, coverage=0.1)

    assert {b.reason for b in breaches} == {
        TripReason.DRAWDOWN, TripReason.THIN_COVERAGE, TripReason.UNEXPECTED_POSITION}


def test_not_measured_is_not_the_same_as_measured_and_fine():
    # A caller with no coverage figure gets no coverage check, rather than a silent pass.
    assert evaluate_tripwires(equity=100.0, peak_equity=100.0, positions={},
                              coverage=None, staleness_days=None) == ()
    assert evaluate_tripwires(equity=100.0, peak_equity=100.0, positions={},
                              coverage=0.0)[0].reason is TripReason.THIN_COVERAGE


def test_stale_data_and_a_nav_mismatch_each_trip():
    stale = evaluate_tripwires(equity=100.0, peak_equity=100.0, positions={},
                               staleness_days=9, max_staleness_days=4)
    assert stale[0].reason is TripReason.STALE_DATA

    mismatch = evaluate_tripwires(equity=100.0, peak_equity=100.0, positions={},
                                  nav_disagreement=-0.04, nav_tolerance=0.005)
    assert mismatch[0].reason is TripReason.NAV_MISMATCH


def test_nothing_trips_on_an_unmeasured_peak():
    # Session one has no high-water mark to measure against, and inventing one from the
    # opening equity would report a 0% drawdown as meaningful.
    assert evaluate_tripwires(equity=1.0, peak_equity=0.0, positions={}) == ()
