"""
The scheduled entry point: refresh the bars, run the session, email the outcome.

    stock-agent-daily                       # fetch, dry-run session, email the result
    stock-agent-daily --submit              # the same, armed to send orders
    stock-agent-daily --submit --strategy equal

Other flags pass through to `stock-agent-session`. Emails after every run, clean ones
included, unless `--quiet-when-clean`. Refuses to run without email settings unless
`--no-email`.
"""

import argparse
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from stock_agent.config import DAILY_LOG_DIR, ROOT
from stock_agent.execution.notify import EmailNotifier, NotifierNotConfigured

FETCH_TIMEOUT_SECONDS = 30 * 60
SESSION_TIMEOUT_SECONDS = 30 * 60
EMAIL_TAIL_LINES = 150

# Operator actions; a kill-switch reset must never be reachable from a schedule.
_REFUSED_PASSTHROUGH = ("--reset-kill-switch", "--status", "--note")


@dataclass(frozen=True)
class StepResult:
    name: str
    returncode: int | None  # None: timed out or could not start
    output: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


Runner = Callable[[str, Sequence[str], int], StepResult]


def run_step(name: str, argv: Sequence[str], timeout: int) -> StepResult:
    # Piped stdout on Windows defaults to the ANSI code page and crashes on non-Latin-1 text.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        done = subprocess.run(list(argv), cwd=ROOT, env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        partial = _text(exc.stdout) + _text(exc.stderr)
        return StepResult(name, None, f"{partial}\n[timed out after {timeout}s]")
    except OSError as exc:
        return StepResult(name, None, f"[could not start: {exc}]")
    return StepResult(name, done.returncode, done.stdout + done.stderr)


def _text(stream) -> str:
    if stream is None:
        return ""
    return stream.decode("utf-8", "replace") if isinstance(stream, bytes) else stream


def run_daily(session_args: Sequence[str], *, notifier: EmailNotifier | None,
              skip_fetch: bool = False, quiet_when_clean: bool = False,
              runner: Runner = run_step, log_dir: Path = DAILY_LOG_DIR,
              now: datetime | None = None) -> int:
    started = now or datetime.now()
    steps: list[StepResult] = []

    if not skip_fetch:
        steps.append(runner("fetch", [sys.executable, "-m", "stock_agent.pipeline",
                                      "--fetch", "--refetch"], FETCH_TIMEOUT_SECONDS))

    # A failed fetch stops the day: the old cache may still pass the staleness guard.
    if all(step.ok for step in steps):
        steps.append(runner("session", [sys.executable, "-m", "stock_agent.execution.cli",
                                        *session_args], SESSION_TIMEOUT_SECONDS))

    status = _status(steps)
    log_path = _write_log(log_dir, started, status, session_args, steps)
    clean = status == "CLEAN"

    if notifier is None or (clean and quiet_when_clean):
        return 0 if clean else 1

    try:
        notifier.send(f"[stock-agent] {status} {started:%Y-%m-%d}",
                      _email_body(status, session_args, steps, log_path))
    except Exception:
        _append(log_path, f"\nALERT EMAIL FAILED\n{traceback.format_exc()}")
        print(f"alert email failed; see {log_path}", file=sys.stderr)
        return 1
    return 0 if clean else 1


def _status(steps: list[StepResult]) -> str:
    if any(step.returncode is None for step in steps):
        return "FAILED"
    if steps[0].name == "fetch" and not steps[0].ok:
        return "FAILED"
    return "CLEAN" if all(step.ok for step in steps) else "ATTENTION"


def _header(status: str, session_args: Sequence[str], steps: list[StepResult]) -> str:
    lines = [f"status:  {status}",
             f"session: stock-agent-session {' '.join(session_args)}".rstrip()]
    for step in steps:
        code = "did not finish" if step.returncode is None else f"exit {step.returncode}"
        lines.append(f"  {step.name:<8} {code}")
    if not any(step.name == "session" for step in steps):
        lines.append("  session  not run (fetch failed)")
    return "\n".join(lines)


def _write_log(log_dir: Path, started: datetime, status: str, session_args: Sequence[str],
               steps: list[StepResult]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{started:%Y-%m-%d}.log"
    body = [f"===== {started:%Y-%m-%d %H:%M:%S} =====", _header(status, session_args, steps)]
    for step in steps:
        body.append(f"\n----- {step.name} -----\n{step.output.rstrip()}")
    _append(path, "\n".join(body) + "\n\n")
    return path


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _email_body(status: str, session_args: Sequence[str], steps: list[StepResult],
                log_path: Path) -> str:
    parts = [_header(status, session_args, steps), f"full log: {log_path}"]
    for step in steps:
        lines = step.output.rstrip().splitlines()
        note = (f" (last {EMAIL_TAIL_LINES} of {len(lines)} lines)"
                if len(lines) > EMAIL_TAIL_LINES else "")
        shown = "\n".join(lines[-EMAIL_TAIL_LINES:])
        parts.append(f"----- {step.name}{note} -----\n{shown}")
    return "\n\n".join(parts) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submit", action="store_true",
                        help="Pass --submit to the session, so it sends orders")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Use the bar cache as it is instead of refreshing it first")
    parser.add_argument("--quiet-when-clean", action="store_true",
                        help="Email only when the run was not clean")
    parser.add_argument("--no-email", action="store_true",
                        help="Run without alerting at all, for testing the schedule")
    args, passthrough = parser.parse_known_args(argv)

    refused = [flag for flag in passthrough if flag.split("=")[0] in _REFUSED_PASSTHROUGH]
    if refused:
        parser.error(f"{', '.join(refused)} is an operator action; run "
                     "stock-agent-session directly for it")

    notifier = None
    if not args.no_email:
        try:
            notifier = EmailNotifier.from_env()
        except NotifierNotConfigured as exc:
            print(f"refusing to run unattended: {exc} (or pass --no-email)", file=sys.stderr)
            return 1

    session_args = [*(["--submit"] if args.submit else []), *passthrough]
    return run_daily(session_args, notifier=notifier, skip_fetch=args.skip_fetch,
                     quiet_when_clean=args.quiet_when_clean)


if __name__ == "__main__":
    sys.exit(main())
