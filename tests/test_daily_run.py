import smtplib
from datetime import datetime

import pytest

from stock_agent.execution import daily
from stock_agent.execution.daily import StepResult, run_daily
from stock_agent.execution.notify import EmailNotifier, NotifierNotConfigured

NOW = datetime(2026, 10, 14, 17, 30)


class RecordingNotifier:
    def __init__(self, fail: bool = False):
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def send(self, subject: str, body: str) -> None:
        if self.fail:
            raise smtplib.SMTPAuthenticationError(535, b"bad credentials")
        self.sent.append((subject, body))


def runner_returning(**codes):
    calls: list[tuple[str, list[str]]] = []

    def run(name, argv, timeout):
        calls.append((name, list(argv)))
        return StepResult(name, codes.get(name, 0), f"{name} output\n")

    run.calls = calls  # type: ignore[attr-defined]
    return run


def run(tmp_path, runner, notifier=None, **kwargs):
    notifier = notifier if notifier is not None else RecordingNotifier()
    code = run_daily(kwargs.pop("session_args", ["--submit"]), notifier=notifier,
                     runner=runner, log_dir=tmp_path, now=NOW, **kwargs)
    return code, notifier


def test_a_clean_day_fetches_then_runs_the_session_and_sends_a_heartbeat(tmp_path):
    runner = runner_returning()
    code, notifier = run(tmp_path, runner)

    assert code == 0
    assert [name for name, _ in runner.calls] == ["fetch", "session"]
    assert runner.calls[1][1][-1] == "--submit"
    [(subject, body)] = notifier.sent
    assert subject == "[stock-agent] CLEAN 2026-10-14"
    assert "session output" in body


def test_quiet_when_clean_suppresses_only_the_clean_email(tmp_path):
    code, notifier = run(tmp_path, runner_returning(), quiet_when_clean=True)
    assert code == 0 and notifier.sent == []

    code, notifier = run(tmp_path, runner_returning(session=1), quiet_when_clean=True)
    assert code == 1 and notifier.sent[0][0].startswith("[stock-agent] ATTENTION")


def test_a_failed_fetch_stops_the_day_before_the_session(tmp_path):
    runner = runner_returning(fetch=1)
    code, notifier = run(tmp_path, runner)

    assert code == 1
    assert [name for name, _ in runner.calls] == ["fetch"]
    subject, body = notifier.sent[0]
    assert subject == "[stock-agent] FAILED 2026-10-14"
    assert "not run (fetch failed)" in body


def test_a_session_that_is_not_clean_is_attention(tmp_path):
    code, notifier = run(tmp_path, runner_returning(session=1))
    assert code == 1
    assert notifier.sent[0][0] == "[stock-agent] ATTENTION 2026-10-14"


def test_a_step_that_never_finished_is_failed_not_attention(tmp_path):
    code, notifier = run(tmp_path, runner_returning(session=None))
    assert code == 1
    assert notifier.sent[0][0].startswith("[stock-agent] FAILED")


def test_skip_fetch_goes_straight_to_the_session(tmp_path):
    runner = runner_returning()
    run(tmp_path, runner, skip_fetch=True)
    assert [name for name, _ in runner.calls] == ["session"]


def test_the_log_is_written_before_the_email_and_survives_a_send_failure(tmp_path):
    code, _ = run(tmp_path, runner_returning(), notifier=RecordingNotifier(fail=True))

    assert code == 1
    log = (tmp_path / "2026-10-14.log").read_text(encoding="utf-8")
    assert "status:  CLEAN" in log
    assert "session output" in log
    assert "ALERT EMAIL FAILED" in log


def test_runs_on_the_same_day_append_to_one_log(tmp_path):
    run(tmp_path, runner_returning())
    run(tmp_path, runner_returning(session=1))
    log = (tmp_path / "2026-10-14.log").read_text(encoding="utf-8")
    assert log.count("=====") == 4


def test_long_output_is_cut_in_the_email_but_not_the_log(tmp_path):
    def chatty(name, argv, timeout):
        return StepResult(name, 0, "\n".join(f"{name} line {i}" for i in range(500)))

    _, notifier = run(tmp_path, chatty)
    body = notifier.sent[0][1]
    assert "session line 499" in body and "session line 0\n" not in body
    assert "last 150 of 500 lines" in body
    assert "session line 0\n" in (tmp_path / "2026-10-14.log").read_text(encoding="utf-8")


def test_the_cli_refuses_to_run_unattended_without_email_settings(monkeypatch):
    monkeypatch.setattr(daily, "load_dotenv", lambda: None)
    for name in ("ALERT_EMAIL_TO", "ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(daily, "run_daily", lambda *a, **k: pytest.fail("must not run"))

    assert daily.main(["--submit"]) == 1


def test_the_cli_passes_unknown_flags_through_and_never_a_kill_switch_reset(monkeypatch):
    monkeypatch.setattr(daily, "load_dotenv", lambda: None)
    seen = {}
    monkeypatch.setattr(daily, "run_daily",
                        lambda args, **kwargs: seen.setdefault("args", args) and 0)
    daily.main(["--no-email", "--submit", "--strategy", "equal"])
    assert seen["args"] == ["--submit", "--strategy", "equal"]

    with pytest.raises(SystemExit):
        daily.main(["--no-email", "--reset-kill-switch", "me"])


def test_a_real_child_process_is_captured_in_utf8(tmp_path):
    import sys

    result = daily.run_step("probe", [sys.executable, "-c", "print('drift → 0')"], 60)
    assert result.ok
    assert "→" in result.output


def test_notifier_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ALERT_EMAIL_TO", "me@example.com")
    monkeypatch.setenv("ALERT_SMTP_USER", "bot@example.com")
    monkeypatch.setenv("ALERT_SMTP_PASSWORD", "app-password")
    monkeypatch.delenv("ALERT_SMTP_HOST", raising=False)
    notifier = EmailNotifier.from_env()

    assert notifier.host == "smtp.gmail.com" and notifier.port == 465
    assert "app-password" not in repr(notifier)

    monkeypatch.delenv("ALERT_SMTP_PASSWORD")
    with pytest.raises(NotifierNotConfigured, match="ALERT_SMTP_PASSWORD"):
        EmailNotifier.from_env()


def test_the_notifier_logs_in_and_sends_one_message(monkeypatch):
    sessions = []

    class FakeSMTP:
        def __init__(self, host, port, timeout, context):
            self.host, self.port, self.messages = host, port, []
            sessions.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            self.login_as = (user, password)

        def send_message(self, message):
            self.messages.append(message)

    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    EmailNotifier(to="me@example.com", user="bot@example.com",
                  password="pw").send("subject", "body")

    [smtp] = sessions
    assert smtp.login_as == ("bot@example.com", "pw")
    [message] = smtp.messages
    assert message["To"] == "me@example.com" and message["Subject"] == "subject"
