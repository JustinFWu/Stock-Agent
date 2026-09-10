"""
Atomic artifact writes.

Every cache in this repo is something the next run depends on: the bars, the fetch
manifest, the derived feature frames, the one saved model. A write interrupted
partway used to leave a truncated file exactly where a good one was expected, and
a partial parquet or JSON is often still parseable as *something*, so the failure
arrives later and looks like a data problem rather than a crashed write.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))
from src.storage import atomic_path


def test_a_completed_write_replaces_the_destination(tmp_path):
    destination = tmp_path / "artifact.txt"
    destination.write_text("old")

    with atomic_path(destination) as tmp:
        tmp.write_text("new")

    assert destination.read_text() == "new"


def test_a_failed_write_leaves_the_previous_artifact_intact(tmp_path):
    """
    The behaviour that matters: a failed rebuild costs you the rebuild, not the
    artifact you already had. Writing in place costs you both.
    """
    destination = tmp_path / "artifact.txt"
    destination.write_text("the good copy")

    with pytest.raises(RuntimeError), atomic_path(destination) as tmp:
        tmp.write_text("half a file")
        raise RuntimeError("interrupted midway")

    assert destination.read_text() == "the good copy"


def test_no_temporary_file_survives_either_outcome(tmp_path):
    """A leftover temporary is a file a later run can mistake for a real artifact."""
    destination = tmp_path / "artifact.txt"

    with atomic_path(destination) as tmp:
        tmp.write_text("fine")
    assert list(tmp_path.iterdir()) == [destination]

    with pytest.raises(RuntimeError), atomic_path(destination) as tmp:
        tmp.write_text("doomed")
        raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == [destination]


def test_the_temporary_sits_beside_its_destination(tmp_path):
    """
    Same directory, because `os.replace` is only atomic within a filesystem.

    A temporary in the system temp directory turns the rename into a copy, which is
    precisely the non-atomic operation being avoided.
    """
    destination = tmp_path / "nested" / "artifact.txt"

    with atomic_path(destination) as tmp:
        assert tmp.parent == destination.parent
        tmp.write_text("x")

    assert destination.read_text() == "x"


def test_parent_directories_are_created(tmp_path):
    destination = tmp_path / "a" / "b" / "artifact.txt"

    with atomic_path(destination) as tmp:
        tmp.write_text("x")

    assert destination.exists()
