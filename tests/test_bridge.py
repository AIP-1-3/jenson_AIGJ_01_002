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
