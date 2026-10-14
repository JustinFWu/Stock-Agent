import sys

import numpy as np
import pandas as pd
from conftest import make_panel

from stock_agent.execution import cli, session


def isolate(monkeypatch, tmp_path):
    real, fake = tmp_path / "state", tmp_path / "state" / "fake"
    monkeypatch.setattr(cli, "ORDER_LOG_PATH", real / "orders.jsonl")
    monkeypatch.setattr(cli, "KILL_SWITCH_PATH", real / "kill_switch.json")
    monkeypatch.setattr(cli, "SESSION_LOG_PATH", real / "sessions.jsonl")
    monkeypatch.setattr(cli, "FAKE_STATE_DIR", fake)
    return real, fake


def test_fake_and_real_brokers_share_no_state_file(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    fake = cli.state_paths("fake")
    for name in ("alpaca-paper", "alpaca-live"):
        real = cli.state_paths(name)
        assert not {fake.orders, fake.kill_switch, fake.sessions} & {
            real.orders, real.kill_switch, real.sessions}


def test_a_fake_session_is_recorded_only_under_the_fake_state(monkeypatch, tmp_path):
    real, fake = isolate(monkeypatch, tmp_path)
    dates = pd.bdate_range(end="2024-06-28", periods=300)
    prices = pd.DataFrame({"AAPL": 100.0 * 1.0003 ** np.arange(len(dates))}, index=dates)
    monkeypatch.setattr(session, "load_price_panel", lambda tickers: make_panel(prices))
    monkeypatch.setattr(sys, "argv", ["stock-agent-session", "--broker", "fake",
                                      "--strategy", "equal"])
    cli.main()

    assert (fake / "sessions.jsonl").exists()
    assert not (real / "sessions.jsonl").exists()
    assert not (real / "orders.jsonl").exists()
