"""
Derived-cache provenance.

A cached feature frame is only trustworthy if it still describes the computation
being asked for. The old check asked one question — does the raw manifest still
name this start date and interval — which almost nothing actually changes:

  * `--fetch --refetch` replaces the bars and leaves every field the old check read
    identical, so training reused features derived from data that no longer existed.
  * Changing `FORWARD_DAYS` changed the labels, not the manifest, so the model
    trained on the old horizon while its saved payload advertised the new one.
  * Changing a feature definition changed nothing observable at all.

The fingerprint closes all three. These tests exercise its sensitivity directly,
because that sensitivity *is* the fix.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.data import dataset
from src.data.fetcher import FetchError

BASE_MANIFEST = {
    "start": "2005-01-01",
    "interval": "1d",
    "rows": 5200,
    "first_bar": "2005-01-03",
    "last_bar": "2026-08-06",
    "fetched_at": "2026-08-07T09:00:00+00:00",
}


@pytest.fixture
def stub_manifest(monkeypatch):
    """Pin the manifest so a fingerprint change can only come from what the test moved."""
    entry = dict(BASE_MANIFEST)
    monkeypatch.setattr(dataset, "manifest_entry", lambda ticker: dict(entry))
    return entry


def test_the_same_inputs_give_the_same_fingerprint(stub_manifest):
    assert dataset._fingerprint("AAPL") == dataset._fingerprint("AAPL")


def test_a_refetch_invalidates_derived_frames(stub_manifest, monkeypatch):
    """
    The headline case. The window is unchanged, so every field the old check read
    still matches — but the bars themselves were replaced, and anything derived from
    them is now describing data that is gone.
    """
    before = dataset._fingerprint("AAPL")

    refetched = dict(stub_manifest, fetched_at="2026-09-05T09:00:00+00:00", rows=5221)
    monkeypatch.setattr(dataset, "manifest_entry", lambda ticker: refetched)

    assert dataset._fingerprint("AAPL") != before


def test_a_changed_history_window_invalidates_derived_frames(stub_manifest, monkeypatch):
    before = dataset._fingerprint("AAPL")

    rewound = dict(stub_manifest, start="1995-01-01")
    monkeypatch.setattr(dataset, "manifest_entry", lambda ticker: rewound)

    assert dataset._fingerprint("AAPL") != before


def test_a_changed_label_horizon_invalidates_derived_frames(stub_manifest, monkeypatch):
    """
    The quietest of the three. Nothing about the raw data moves, so the old check saw
    a current cache — and the model trained on five-day labels while its saved payload
    said ten.
    """
    before = dataset._fingerprint("AAPL")
    monkeypatch.setattr(dataset, "FORWARD_DAYS", 10)
    assert dataset._fingerprint("AAPL") != before


def test_changed_feature_parameters_invalidate_derived_frames(stub_manifest, monkeypatch):
    for attribute, value in (("RV_WINDOWS", [1, 5, 21]),
                             ("EWMA_LAMBDA", 0.97),
                             ("ATR_PERIOD", 20),
                             ("RETURN_HORIZONS", [1, 5, 21])):
        before = dataset._fingerprint("AAPL")
        with monkeypatch.context() as m:
            m.setattr(dataset, attribute, value)
            assert dataset._fingerprint("AAPL") != before, f"{attribute} did not invalidate"


def test_the_schema_version_invalidates_derived_frames(stub_manifest, monkeypatch):
    """
    The escape hatch for changes no parameter records — a redefined feature, a new
    column. Without it, editing a builder leaves every cached frame looking current.
    """
    before = dataset._fingerprint("AAPL")
    monkeypatch.setattr(dataset, "FEATURE_SCHEMA_VERSION", dataset.FEATURE_SCHEMA_VERSION + 1)
    assert dataset._fingerprint("AAPL") != before


def test_an_uncached_ticker_fingerprints_differently_from_a_cached_one(monkeypatch):
    monkeypatch.setattr(dataset, "manifest_entry", lambda ticker: {})
    unknown = dataset._fingerprint("NOPE")

    monkeypatch.setattr(dataset, "manifest_entry", lambda ticker: dict(BASE_MANIFEST))
    assert dataset._fingerprint("NOPE") != unknown


def test_only_data_access_failures_are_skippable():
    """
    A ticker that cannot be read is a data problem and a reason to move on. A
    TypeError from a feature builder is a bug in this repo, and swallowing it would
    train the model on whichever names happened to dodge it — with a reassuring
    'stacked 79/82' in the log.
    """
    for expected in (FetchError, FileNotFoundError, KeyError, ValueError):
        assert issubclass(expected, dataset.SKIPPABLE)

    for defect in (TypeError, AttributeError, NameError, ZeroDivisionError):
        assert not issubclass(defect, dataset.SKIPPABLE)
