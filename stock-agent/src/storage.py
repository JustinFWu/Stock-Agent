"""
Durable writes for the artifacts everything else reads.

Every cache in this repo is written straight to its final path: the parquet bars,
the fetch manifest, the derived feature frames, the saved model. A write that is
interrupted partway — Ctrl-C during a twenty-minute fetch, a full disk, a crash —
leaves a truncated file sitting exactly where the next run expects a good one.
Worse, a reader can open the file *during* the write and get half of it, with no
error, because a partial parquet or a partial JSON is often still parseable as
something.

The fix is the standard one: build the new file under a temporary name in the same
directory, then rename it over the destination. `os.replace` is atomic on POSIX and
on Windows, so a reader sees either the complete old file or the complete new one
and never a state in between. Same directory matters — a rename across filesystems
is a copy, and a copy is exactly the non-atomic operation being avoided.

This does not make concurrent *writers* safe. Two processes fetching the same
ticker still race, and the last rename wins; see `fetcher` for the read-modify-write
on the manifest, which is the place that actually loses data when it happens.
"""

import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(destination: Path):
    """
    Yield a temporary path to write, then move it into place on a clean exit.

    On any exception the temporary file is removed and the destination is left
    exactly as it was, which is the behaviour that matters: a failed rebuild
    should cost you the rebuild, not the artifact you already had.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")

    try:
        yield tmp
        os.replace(tmp, destination)
    finally:
        # Only reached with the file still present if the body raised, or if the
        # replace itself failed. Either way the temporary is not something a later
        # run should find lying around and mistake for a real artifact.
        if tmp.exists():
            tmp.unlink()
