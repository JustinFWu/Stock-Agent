import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(destination: Path):
    # An interrupted write must not leave a truncated file where the next run expects a
    # good one. The temp file shares the destination's directory because a
    # cross-filesystem rename degrades to a copy, which is not atomic.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")

    try:
        yield tmp
        os.replace(tmp, destination)
    finally:
        # Present here only if the body or the replace raised; a later run must not find
        # it and mistake it for a real artifact. Concurrent writers still race — last
        # rename wins, see fetcher's read-modify-write on the manifest.
        if tmp.exists():
            tmp.unlink()
