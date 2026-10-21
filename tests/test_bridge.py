import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from test_service import CFG, NOW, _engine, _raw

from tsinfer.bridge.__main__ import to_line
from tsinfer.bridge.points import to_points


def _payload(space="lab", **over):
    """서비스가 실제로 MQTT 로 내는 모양 (publisher.publish 와 같은 조립)."""
    out = _engine().run_once(NOW, _raw(space=space))
    body = out["spaces"][space]
    return {"schema_version": out["schema_version"], "generated_at": out["generated_at"],
            "model": out["model"], "space": space, **body, **over}


def test_result_point_has_current_forecast_and_flags():
    pts = to_points(_payload())
    res = pts[0]
    assert res.measurement == "tsinfer_result"
    assert res.tags == {"space": "lab", "model": res.tags["model"]}
    assert res.fields["data_ok"] == 1
    assert {"current_co2", "fc_co2_60", "fc_temperature_5", "is_anomaly", "alert_count"} <= res.fields.keys()
    assert all(isinstance(v, (int, float)) for v in res.fields.values())


def test_forecast_points_are_shifted_to_target_time():
    payload = _payload()
    gen = datetime.fromisoformat(payload["generated_at"])
    fc = [p for p in to_points(payload) if p.measurement == "tsinfer_forecast"]
    # 보고 지평 4개 × CO₂·온도
    assert sorted({p.tags["horizon"] for p in fc}, key=int) == ["5", "15", "30", "60"]
    assert len(fc) == 8
    p60 = next(p for p in fc if p.tags["horizon"] == "60" and "co2" in p.fields)
    assert p60.time == gen + timedelta(minutes=60)
    assert p60.fields["co2"] == payload["forecast"]["co2"]["60"]


def test_null_state_and_no_data_are_not_written_as_zero():
    payload = _payload(data_ok=False, current={}, forecast={}, co2_ventilate_in_minutes=None,
                       state={"occupied": None, "ventilation_alert": False},
                       anomaly={"is_anomaly": True, "alerts": [
                           {"type": "data_gap", "metric": None, "severity": "high", "detail": "15분 수신 없음"}]})
    pts = to_points(payload)
    res = pts[0]
    assert res.fields["data_ok"] == 0
    assert "occupied" not in res.fields            # 판단 불가는 0(빈 방)이 아니라 빈 값
    assert res.fields["ventilation_alert"] == 0
    assert res.fields["alert_high_count"] == 1
    alert = [p for p in pts if p.measurement == "tsinfer_alert"]
    assert alert[0].tags["type"] == "data_gap" and alert[0].tags["metric"] == "-"
    assert not [p for p in pts if p.measurement == "tsinfer_forecast"]


def test_line_protocol_escapes_korean_and_spaces():
    pts = to_points(_payload(space="lab"))
    pts[0].tags["space"] = "사무실 밖"
    line = to_line(pts[0])
    assert line.startswith("tsinfer_result,model=")
    assert "space=사무실\\ 밖" in line
    assert line.rsplit(" ", 1)[1].isdigit()        # 초 단위 시각


def test_dashboards_reference_bridge_measurements():
    """대시보드 쿼리와 브리지가 쓰는 이름이 어긋나지 않게 한다."""
    root = Path(__file__).resolve().parents[1] / "grafana" / "dashboards"
    text = (root / "tsinfer-live.json").read_text(encoding="utf-8")
    json.loads(text)
    for name in ("tsinfer_result", "tsinfer_forecast", "tsinfer_alert", "fc_co2_60", "ventilation_alert"):
        assert name in text
    used = set(re.findall(r'tsinfer_result\\" and r._field == \\"(\w+)', text))
    assert used, "대시보드에서 tsinfer_result 필드를 못 찾음"
    produced = set(to_points(_payload())[0].fields)
    optional = {"co2_ventilate_in_minutes", "occupied"}   # 값이 null 이면 안 쓰는 필드
    assert used <= produced | optional, used - produced - optional


def test_mirror_copies_new_collector_rows_in_iot_bucket_shape(tmp_path, monkeypatch):
    """학교 InfluxDB 에 못 닿을 때: 수집기 SQLite → 로컬 InfluxDB 를 iot-bucket 과 같은 모양으로."""
    from tsinfer.bridge import mirror as m
    from tsinfer.collector.mapping import Reading
    from tsinfer.collector.store import Store

    db = tmp_path / "t.db"
    store = Store(db)
    t0 = datetime.now(timezone.utc) - timedelta(minutes=5)
    store.write([Reading(space="사무실 밖", device="EM300", metric="temperature", value=21.5,
                         sensor_ts=t0, received_ts=t0, source="chirpstack_env")])

    class Sink:
        lines: list[str] = []

        def write(self, lines):
            self.lines.extend(lines)

    sink = Sink()
    mir = m.SensorMirror(db, sink, backfill_hours=1)
    assert mir.step() == 1
    assert sink.lines[0] == f"temperature,device_id=EM300,location=사무실\\ 밖 value=21.5 {int(t0.timestamp())}"
    assert mir.step() == 0                         # 이미 옮긴 행은 다시 안 옮김
    t1 = t0 + timedelta(minutes=1)
    store.write([Reading(space="실습실", device="AM107", metric="co2", value=800,
                         sensor_ts=t1, received_ts=t1, source="chirpstack_env")])
    assert mir.step() == 1 and sink.lines[-1].startswith("co2,device_id=AM107,location=실습실 value=800.0 ")


def test_mirror_refuses_to_write_back_to_source_server(monkeypatch):
    from tsinfer.bridge.__main__ import mirror_enabled

    monkeypatch.setenv("BRIDGE_MIRROR_SENSORS", "1")
    monkeypatch.setenv("INFLUX_URL", "http://school:8086")
    monkeypatch.setenv("INFLUX_RESULT_URL", "http://school:8086/")
    import pytest

    with pytest.raises(RuntimeError):
        mirror_enabled()
    monkeypatch.setenv("INFLUX_RESULT_URL", "http://localhost:8186")
    assert mirror_enabled()


def test_sink_writes_with_second_precision(monkeypatch):
    """줄 끝 시각이 초 단위이므로 쓰기 정밀도도 초여야 한다 (ns 로 읽히면 1970년 점이 돼 거부됨)."""
    from tsinfer.bridge.__main__ import InfluxSink

    calls = []

    class FakeWrite:
        def write(self, **kw):
            calls.append(kw)

    sink = InfluxSink.__new__(InfluxSink)
    sink.bucket, sink.org, sink.write_api = "tsinfer", "org", FakeWrite()
    from collections import deque

    sink.pending = deque(maxlen=10)
    sink.write([to_line(to_points(_payload())[0])])
    assert calls and all(str(c["write_precision"]).lower().endswith("s") for c in calls)
    assert not sink.pending


def test_dashboard_maps_do_not_emit_extra_time_columns():
    """map 으로 새 레코드를 만들 때 _start·_stop 을 일반 열로 남기면 Grafana 가 그것을 시간축으로 써서
    선이 하나도 안 그려진다 (범례 Mean·Max 만 나옴, 2026-10-21 Jetson 60분 오차 패널)."""
    root = Path(__file__).resolve().parents[1] / "grafana" / "dashboards"
    d = json.loads((root / "tsinfer-live.json").read_text(encoding="utf-8"))
    queries = [t["query"] for p in d["panels"] for t in p.get("targets", [])]
    explicit = [ln for q in queries for ln in q.splitlines() if "map(fn: (r) => ({_time" in ln]
    assert explicit, "오차 패널의 map 을 못 찾음"
    assert not [ln for ln in explicit if "_start" in ln or "_stop" in ln]
