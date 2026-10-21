from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from tsinfer.service.app import Runner, create_app
from tsinfer.service.monitor import SelfMonitor
from tsinfer.service.summary import summarize

CFG = {"preprocess": {"resample": "5min"}}
T0 = datetime(2026, 10, 21, 9, 0, 5, tzinfo=timezone.utc)


def _result(now, cur, pred60, space="lab", data_ok=True):
    return {"generated_at": now.isoformat(), "model": "fake",
            "spaces": {space: {"data_ok": data_ok, "current": {"co2": cur}, "anomaly": {"alerts": []},
                               "forecast": {"co2": {"60": pred60}}}}}


class FakeHook:
    def __init__(self):
        self.msgs = []

    def __call__(self, text):
        self.msgs.append(text)
        return True


def _drive(mon, minutes, truth, pred_offset):
    """매분 한 주기. 실제값 truth(분), 60분 예측 = 60분 뒤 실제값 + pred_offset."""
    sent = 0
    for m in range(minutes):
        now = T0 + timedelta(minutes=m)
        sent += mon.observe(_result(now, truth(m), truth(m + 60) + pred_offset))
    return sent


def test_scores_after_60_minutes_and_alerts_once_per_hour():
    hook = FakeHook()
    mon = SelfMonitor(CFG, notify=hook, min_n=3)

    def rising(m):        # 5분 칸 단위로 오르는 실제값 (엔진의 current 처럼 칸마다 같은 값)
        return 600.0 + 10 * (m // 5)

    _drive(mon, 60, rising, 0.0)
    assert mon.stats("lab")["n"] == 0                    # 60분 뒤 칸이 완성되기 전에는 채점하지 않는다
    mon2 = SelfMonitor(CFG, notify=hook, min_n=3)
    _drive(mon2, 100, rising, 0.0)
    s = mon2.stats("lab")
    assert s["n"] >= 6 and s["mae"] == 0.0 and s["mae_persistence"] == 120.0
    assert hook.msgs == []                               # 완벽 예측 → 알림 없음

    bad = SelfMonitor(CFG, notify=hook, min_n=3, realert=timedelta(hours=1))
    sent = _drive(bad, 130, rising, 300.0)               # 300ppm 틀림 → 경고
    assert bad.stats("lab")["mae"] == 300.0 and bad.snapshot()["spaces"]["lab"]["alerting"]
    assert sent == 1                                     # 계속 나빠도 같은 알림은 1시간에 한 번
    sent += _drive_from(bad, 130, 190, rising, 300.0)
    assert sent == 2 and len(hook.msgs) == 2


def _drive_from(mon, start, end, truth, off):
    sent = 0
    for m in range(start, end):
        sent += mon.observe(_result(T0 + timedelta(minutes=m), truth(m), truth(m + 60) + off))
    return sent


def test_without_webhook_skips_notify_and_env_off_returns_none(monkeypatch):
    monkeypatch.delenv("SELF_MONITOR", raising=False)
    assert SelfMonitor.from_env(CFG) is None
    monkeypatch.setenv("SELF_MONITOR", "1")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "")
    mon = SelfMonitor.from_env(CFG)
    assert mon is not None and mon.notify is None
    sent = _drive(mon, 150, lambda m: 600.0, 500.0)
    assert sent == 0 and mon.snapshot()["spaces"]["lab"]["alerting"]


def test_stale_predictions_expire():
    mon = SelfMonitor(CFG, notify=None)
    mon.observe(_result(T0, 600.0, 700.0))
    mon.observe(_result(T0 + timedelta(minutes=90), 600.0, None, data_ok=False))
    assert mon.snapshot()["pending"] == 0 and mon.expired == 1


def test_runner_health_and_summary_include_monitor(tmp_path):
    class FakeEngine:
        def run_once(self, now=None):
            return _result(now, 600.0, 800.0)

    log = tmp_path / "c.jsonl"
    hook = FakeHook()
    runner = Runner(FakeEngine(), cycle_log=log, monitor=SelfMonitor(CFG, notify=hook, min_n=1))
    client = TestClient(create_app(runner, start_loop=False))
    for m in range(70):
        runner.cycle(T0 + timedelta(minutes=m))
    h = client.get("/health").json()
    assert h["monitor"]["spaces"]["lab"]["mae"] == 200.0
    s = summarize(log, hours=2, now=T0 + timedelta(minutes=70))
    assert s["monitor"]["latest"]["lab"]["mae"] == 200.0 and s["monitor"]["alerts_sent"] == len(hook.msgs) == 1

    # 모니터 없으면 /health·요약에 monitor 필드가 없다 (지금 동작 그대로)
    log2 = tmp_path / "d.jsonl"
    plain = Runner(FakeEngine(), cycle_log=log2)
    plain.cycle(T0)
    assert "monitor" not in TestClient(create_app(plain, start_loop=False)).get("/health").json()
    assert "monitor" not in summarize(log2, hours=1, now=T0 + timedelta(minutes=1))
