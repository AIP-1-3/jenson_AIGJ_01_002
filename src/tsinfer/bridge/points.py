"""서비스 결과(MQTT 페이로드, docs/interface.md schema 1.0) → InfluxDB 점.

Grafana 가 실시간 센서(iot-bucket)와 같은 화면에서 추론 결과를 그리려면 결과도 시계열 DB 에 있어야 한다.
서비스 코드는 건드리지 않고, 서비스가 이미 내보내는 MQTT 결과를 받아 바꾸기만 한다.

measurement 3개 (태그: space, model)
- tsinfer_result   시각 = generated_at. 현재값, 5~60분 예측값, 경보·상태 (0/1), data_ok
- tsinfer_forecast 시각 = generated_at + h분 (예측이 가리키는 시각), 태그 horizon.
                   실제 CO₂ 와 같은 시간축에 겹쳐 그리면 '60분 전에 한 예측 vs 실제' 를 바로 볼 수 있다.
- tsinfer_alert    경보 1건 = 1점, 태그 type·metric·severity, 필드 detail
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

FORECAST_METRICS = ("co2", "temperature")


@dataclass
class Point:
    measurement: str
    time: datetime
    tags: dict[str, str]
    fields: dict[str, float | int | str] = field(default_factory=dict)


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _flag(x: Any) -> int | None:
    """bool → 0/1. null(판단 불가)은 점을 비워 Grafana 에서 '값 없음' 으로 보이게 한다."""
    return None if x is None else int(bool(x))


def to_points(payload: dict[str, Any]) -> list[Point]:
    gen = datetime.fromisoformat(str(payload["generated_at"]).replace("Z", "+00:00"))
    tags = {"space": str(payload["space"]), "model": str(payload.get("model", "unknown"))}
    fields: dict[str, float | int | str] = {"data_ok": int(bool(payload.get("data_ok")))}

    for metric, v in (payload.get("current") or {}).items():
        if (x := _num(v)) is not None:
            fields[f"current_{metric}"] = x

    reported = {str(h) for h in payload.get("horizons_reported") or []}
    points: list[Point] = []
    for metric in FORECAST_METRICS:
        for h, v in ((payload.get("forecast") or {}).get(metric) or {}).items():
            if (x := _num(v)) is None:
                continue
            fields[f"fc_{metric}_{h}"] = x
            # 겹쳐 그리기용은 보고 지평(5·15·30·60분)만 남겨 점 수를 줄인다
            if not reported or str(h) in reported:
                points.append(Point("tsinfer_forecast", gen + timedelta(minutes=int(h)),
                                    {**tags, "horizon": str(h)}, {metric: x}))

    vent_in = _num(payload.get("co2_ventilate_in_minutes"))
    if vent_in is not None:
        fields["co2_ventilate_in_minutes"] = vent_in
    state = payload.get("state") or {}
    for key in ("occupied", "ventilation_alert"):
        if (f := _flag(state.get(key))) is not None:
            fields[key] = f

    anomaly = payload.get("anomaly") or {}
    alerts = anomaly.get("alerts") or []
    fields["is_anomaly"] = int(bool(anomaly.get("is_anomaly")))
    fields["alert_count"] = len(alerts)
    fields["alert_high_count"] = sum(1 for a in alerts if a.get("severity") == "high")
    for a in alerts:
        points.append(Point("tsinfer_alert", gen, {
            **tags,
            "type": str(a.get("type", "unknown")),
            "metric": str(a.get("metric") or "-"),
            "severity": str(a.get("severity", "-")),
        }, {"value": 1, "detail": str(a.get("detail", ""))}))

    # tsinfer_forecast 와 시각이 겹쳐도 measurement 가 달라 덮어쓰지 않는다
    points.insert(0, Point("tsinfer_result", gen, tags, fields))
    return points
