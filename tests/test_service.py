import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from tsinfer.collector.mapping import Reading
from tsinfer.collector.store import Store
from tsinfer.service import rules
from tsinfer.service.app import Runner, create_app
from tsinfer.service.engine import Engine, EngineConfig
from tsinfer.service.predictors import Persistence, TrendBaseline
from tsinfer.service.summary import summarize

CFG = {
    "spaces": ["lab", "office"],
    "metrics": ["co2", "temperature", "humidity"],
    "preprocess": {"resample": "5min", "short_gap_max_minutes": 10,
                   "valid_range": {"co2": [350, 5000], "temperature": [-10, 50], "humidity": [0, 100]}},
    "forecast": {"horizon_minutes": 60, "step_minutes": 5, "report_horizons": [5, 15, 30, 60]},
    "thresholds": {"co2_ventilate_ppm": 1000, "co2_critical_ppm": 1500, "ventilation_alert_within_minutes": 30},
}
NOW = datetime(2026, 10, 20, 12, 3, 10, tzinfo=timezone.utc)


def _raw(minutes=180, co2_slope=5.0, end=NOW, space="lab", stale_after=None):
    """1분 간격 원본. co2 는 분당 co2_slope 만큼 오른다."""
    ts = pd.date_range(end=end - timedelta(seconds=10), periods=minutes, freq="1min", tz="UTC")
    if stale_after is not None:
        ts = ts[ts <= end - stale_after]
    n = len(ts)
    rows = []
    for metric, vals in (("co2", 600 + co2_slope * np.arange(n)),
                         ("temperature", np.full(n, 23.0) + 0.01 * np.arange(n)),
                         ("humidity", 40 + np.sin(np.arange(n) / 10))):
        rows.append(pd.DataFrame({"ts": ts, "space": space, "device": "d1", "metric": metric,
                                  "value": vals, "received_ts": ts}))
    return pd.concat(rows, ignore_index=True)


def _engine(forecaster=None, db=None):
    return Engine(CFG, EngineConfig(db_path=db), forecaster or TrendBaseline(CFG))


def test_trend_forecast_rises_and_triggers_ventilation_alert():
    # 마지막 완성 칸 ≈ 600 + 5*~175 → 1,000 근처 아래에서 시작하도록 기울기를 맞춘다
    raw = _raw(minutes=180, co2_slope=2.0)
    out = _engine().run_once(NOW, raw)
    lab = out["spaces"]["lab"]
    fc = lab["forecast"]["co2"]
    assert lab["data_ok"] and set(fc) == {str(h) for h in range(5, 61, 5)}
    assert fc["60"] > fc["5"] > lab["current"]["co2"]
    assert lab["co2_ventilate_in_minutes"] is not None
    assert lab["state"]["occupied"] is True
    # 데이터가 없는 공간은 끊김 경보
    assert out["spaces"]["office"]["data_ok"] is False
    assert out["spaces"]["office"]["anomaly"]["alerts"][0]["type"] == "data_gap"


def test_incomplete_last_bin_is_excluded():
    out = _engine(Persistence(CFG)).run_once(NOW, _raw(co2_slope=1.0))
    # NOW=12:03 → 12:00 칸은 미완성, 11:55 칸이 마지막. 11:55~11:59 평균 ≈ 마지막 값보다 작다
    last_raw = 600 + 1.0 * 179
    assert out["spaces"]["lab"]["current"]["co2"] < last_raw - 3


def test_stale_data_gives_gap_alert_and_no_forecast():
    raw = _raw(stale_after=timedelta(minutes=15))
    lab = _engine().run_once(NOW, raw)["spaces"]["lab"]
    assert lab["data_ok"] is False and lab["forecast"] == {}
    assert lab["anomaly"]["is_anomaly"] and lab["anomaly"]["alerts"][0]["type"] == "data_gap"


def test_stuck_and_out_of_range_rules():
    raw = _raw(co2_slope=0.0)
    raw.loc[raw.index[-1], "value"] = 99.0  # 마지막 humidity → 범위 밖
    lab = _engine().run_once(NOW, raw)["spaces"]["lab"]
    kinds = {(a["type"], a["metric"]) for a in lab["anomaly"]["alerts"]}
    assert ("stuck", "co2") in kinds
    assert rules.occupied(pd.Series([600.0, 600, 600, 600])) is False


def test_engine_reads_collector_db(tmp_path):
    db = tmp_path / "t.db"
    store = Store(db)
    raw = _raw(minutes=60)
    store.write(Reading(r.space, r.device, r.metric, float(r.value), r.ts.to_pydatetime(),
                        r.received_ts.to_pydatetime(), "test") for r in raw.itertuples())
    out = _engine(db=db).run_once(NOW)
    assert out["spaces"]["lab"]["data_ok"] and "co2" in out["spaces"]["lab"]["forecast"]


def test_runner_api_and_summary(tmp_path):
    class FakeEngine:
        def __init__(self):
            self.calls = 0

        def run_once(self, now=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return _engine().run_once(NOW, _raw())

    log = tmp_path / "cycles.jsonl"
    runner = Runner(FakeEngine(), cycle_log=log)
    client = TestClient(create_app(runner, start_loop=False))
    assert client.get("/latest").status_code == 503

    runner.cycle(NOW)
    body = client.get("/latest/lab").json()
    assert body["space"] == "lab" and "forecast" in body
    assert client.get("/latest/nowhere").status_code == 404
    assert client.get("/health").json()["ok"] is True

    runner.cycle(NOW + timedelta(minutes=1))           # 실패해도 마지막 결과는 유지
    h = client.get("/health").json()
    assert h["ok"] is False and h["consecutive_failures"] == 1 and h["last_generated_at"]

    s = summarize(log, hours=1, now=NOW + timedelta(minutes=2))
    assert s["cycles"] == 2 and s["failed_cycles"] == 1 and s["data_gap_minutes"]["office"] == 1
    assert all(json.loads(line)["duration_ms"] >= 0 for line in log.read_text().splitlines())
