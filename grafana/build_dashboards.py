"""Grafana 대시보드 JSON 생성: python grafana/build_dashboards.py → grafana/dashboards/tsinfer-live.json

JSON 을 손으로 고치면 패널 id·위치가 쉽게 어긋나서, 쿼리와 배치를 이 파일에서 만들고 결과 JSON 을 커밋한다.
- 데이터 원본 2개: ${bucket}(iot-bucket, 센서 원본, 읽기만) / ${result_bucket}(tsinfer, 브리지가 쓰는 추론 결과)
- 데이터 소스는 변수(ds)로 받아, 학교 Grafana 에 가져오기(import)와 Jetson Grafana 프로비저닝 둘 다 그대로 쓴다.
- 쿼리 언어: InfluxDB 2.x Flux (influx_export.py 와 같은 measurement·field·태그 구조)
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "dashboards" / "tsinfer-live.json"
DS = {"type": "influxdb", "uid": "${ds}"}
SPACES = "contains(value: r.location, set: ${space:json})"
RANGE = "|> range(start: v.timeRangeStart, stop: v.timeRangeStop)"

_id = 0
_y = 0


def nid() -> int:
    global _id
    _id += 1
    return _id


def target(query: str, ref: str = "A") -> dict:
    return {"refId": ref, "datasource": DS, "query": query.strip()}


def sensor(measurement: str, fn: str = "mean") -> str:
    """공간별 평균 (같은 공간 센서 여러 대는 평균 — 학습 전처리와 같은 방식)."""
    return f"""
from(bucket: "${{bucket}}")
  {RANGE}
  |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "value" and {SPACES})
  |> map(fn: (r) => ({{r with _value: float(v: r._value)}}))
  |> group(columns: ["location"])
  |> aggregateWindow(every: v.windowPeriod, fn: {fn}, createEmpty: false)
"""


def last_sensor(measurement: str) -> str:
    return f"""
from(bucket: "${{bucket}}")
  |> range(start: -30m)
  |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "value" and {SPACES})
  |> map(fn: (r) => ({{r with _value: float(v: r._value)}}))
  |> group(columns: ["location", "device_id"])
  |> last()
  |> group(columns: ["location"])
  |> mean()
"""


def result_field(field: str, every: str = "v.windowPeriod", fn: str = "last") -> str:
    return f"""
from(bucket: "${{result_bucket}}")
  {RANGE}
  |> filter(fn: (r) => r._measurement == "tsinfer_result" and r._field == "{field}" and contains(value: r.space, set: ${{space:json}}))
  |> group(columns: ["space"])
  |> aggregateWindow(every: {every}, fn: {fn}, createEmpty: false)
"""


def actual_co2(var: str = "actual", space_filter: str = SPACES) -> str:
    return f"""{var} = from(bucket: "${{bucket}}")
  {RANGE}
  |> filter(fn: (r) => r._measurement == "co2" and r._field == "value" and {space_filter})
  |> map(fn: (r) => ({{r with _value: float(v: r._value), space: r.location}}))
  |> group(columns: ["space"])
  |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)"""


def place(w: int, h: int, x: int) -> dict:
    return {"h": h, "w": w, "x": x, "y": _y}


def row(title: str) -> dict:
    global _y
    p = {"type": "row", "id": nid(), "title": title, "collapsed": False, "panels": [],
         "gridPos": {"h": 1, "w": 24, "x": 0, "y": _y}}
    _y += 1
    return p


def advance(h: int) -> None:
    global _y
    _y += h


def thresholds(*steps) -> dict:
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in steps]}


CO2_STEPS = thresholds((None, "green"), (1000, "orange"), (1500, "red"))
BY_SPACE = "${__field.labels.location}"


def stat(title, targets, grid, unit, steps=None, desc="", decimals=None, mappings=None):
    defaults = {"unit": unit, "displayName": BY_SPACE, "thresholds": steps or thresholds((None, "blue"))}
    if decimals is not None:
        defaults["decimals"] = decimals
    if mappings:
        defaults["mappings"] = mappings
    return {"type": "stat", "id": nid(), "title": title, "description": desc, "datasource": DS,
            "gridPos": grid, "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                        "colorMode": "background", "graphMode": "none", "textMode": "value_and_name",
                        "justifyMode": "auto", "orientation": "horizontal"}}


def timeseries(title, targets, grid, unit, desc="", display=BY_SPACE, steps=None, calcs=None,
               draw="line", extra_overrides=None, repeat=None):
    custom = {"drawStyle": draw, "lineWidth": 2, "fillOpacity": 0 if draw == "line" else 60,
              "spanNulls": 600000, "showPoints": "never", "pointSize": 4}
    if steps:
        custom["thresholdsStyle"] = {"mode": "dashed"}
    p = {"type": "timeseries", "id": nid(), "title": title, "description": desc, "datasource": DS,
         "gridPos": grid, "targets": targets,
         "fieldConfig": {"defaults": {"unit": unit, "displayName": display, "custom": custom,
                                      "thresholds": steps or thresholds((None, "green"))},
                         "overrides": extra_overrides or []},
         "options": {"legend": {"showLegend": True, "displayMode": "table" if calcs else "list",
                                "placement": "bottom", "calcs": calcs or []},
                     "tooltip": {"mode": "multi", "sort": "none"}}}
    if repeat:
        p.update({"repeat": repeat, "repeatDirection": "h", "maxPerRow": 3})
    return p


def state_timeline(title, field, grid, mappings, desc=""):
    return {"type": "state-timeline", "id": nid(), "title": title, "description": desc, "datasource": DS,
            "gridPos": grid, "targets": [target(result_field(field))],
            "fieldConfig": {"defaults": {"displayName": "${__field.labels.space}", "mappings": mappings,
                                         "color": {"mode": "thresholds"},
                                         "thresholds": thresholds((None, "green"))}, "overrides": []},
            "options": {"showValue": "never", "mergeValues": True, "rowHeight": 0.8,
                        "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}}}


def onoff(off_text, on_text, off="green", on="red"):
    return [{"type": "value", "options": {"0": {"text": off_text, "color": off, "index": 0},
                                          "1": {"text": on_text, "color": on, "index": 1}}}]


def build() -> dict:
    panels = []

    # ── 1. 실시간 센서 ──────────────────────────────────────────────
    panels.append(row("실시간 센서 — MQTT → InfluxDB ${bucket}"))
    panels += [
        stat("현재 CO₂ (최근 30분 마지막 값, 공간 평균)", [target(last_sensor("co2"))], place(8, 5, 0),
             "suffix: ppm", CO2_STEPS, "1,000ppm 환기 권장(주황), 1,500ppm 위험(빨강). configs/default.yaml thresholds 와 같다.", 0),
        stat("현재 온도", [target(last_sensor("temperature"))], place(8, 5, 8), "celsius",
             thresholds((None, "blue"), (18, "green"), (26, "orange"), (28, "red")), decimals=1),
        stat("현재 습도", [target(last_sensor("humidity"))], place(8, 5, 16), "humidity",
             thresholds((None, "orange"), (30, "green"), (60, "orange")), decimals=0),
    ]
    advance(5)
    panels.append(timeseries("CO₂ (공간별 평균)", [target(sensor("co2"))], place(24, 9, 0), "suffix: ppm",
                             "점선: 1,000 / 1,500ppm", steps=CO2_STEPS, calcs=["lastNotNull", "max", "mean"]))
    advance(9)
    panels += [
        timeseries("온도", [target(sensor("temperature"))], place(12, 8, 0), "celsius", calcs=["lastNotNull", "min", "max"]),
        timeseries("습도", [target(sensor("humidity"))], place(12, 8, 12), "humidity", calcs=["lastNotNull", "min", "max"]),
    ]
    advance(8)

    # 데이터 흐름: 센서별 마지막 수신과 5분당 수신 건수 (끊김·값 고정을 눈으로 확인)
    flow_last = f"""
from(bucket: "${{bucket}}")
  |> range(start: -24h)
  |> filter(fn: (r) => (r._measurement == "co2" or r._measurement == "temperature" or r._measurement == "humidity") and r._field == "value" and {SPACES})
  |> group(columns: ["location", "device_id", "_measurement"])
  |> last()
  |> map(fn: (r) => ({{location: r.location, device_id: r.device_id, metric: r._measurement, value: float(v: r._value), last_seen: r._time,
      minutes_ago: float(v: int(v: now()) - int(v: r._time)) / 60000000000.0}}))
  |> group()
  |> sort(columns: ["minutes_ago"], desc: true)
"""
    panels.append({
        "type": "table", "id": nid(), "title": "센서별 마지막 수신 (오래된 순)", "datasource": DS,
        "description": "minutes_ago 가 10분을 넘으면 서비스가 data_gap 경보를 낸다 (docs/interface.md).",
        "gridPos": place(12, 9, 0), "targets": [target(flow_last)],
        "fieldConfig": {"defaults": {}, "overrides": [
            {"matcher": {"id": "byName", "options": "minutes_ago"}, "properties": [
                {"id": "unit", "value": "m"}, {"id": "decimals", "value": 1},
                {"id": "custom.cellOptions", "value": {"type": "color-background"}},
                {"id": "thresholds", "value": thresholds((None, "green"), (10, "orange"), (60, "red"))}]},
            {"matcher": {"id": "byName", "options": "value"}, "properties": [{"id": "decimals", "value": 1}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"}})
    flow_count = f"""
from(bucket: "${{bucket}}")
  {RANGE}
  |> filter(fn: (r) => (r._measurement == "co2" or r._measurement == "temperature") and r._field == "value" and {SPACES})
  |> group(columns: ["location"])
  |> aggregateWindow(every: 5m, fn: count, createEmpty: true)
"""
    panels.append(timeseries("수신 건수 (5분당, CO₂+온도)", [target(flow_count)], place(12, 9, 12), "short",
                             "0 으로 떨어지면 수집 끊김. 센서 업링크 간격이 바뀌어도 여기서 바로 보인다.", draw="bars"))
    advance(9)

    # ── 2. 추론 결과 ────────────────────────────────────────────────
    panels.append(row("추론 결과 — 서비스 MQTT → 브리지 → InfluxDB ${result_bucket}"))
    fc_vs_actual = [
        target(actual_co2("actual", 'r.location == "${space}"') + """
actual
  |> set(key: "series", value: "실제")
  |> group(columns: ["series"])""", "A"),
        target(f"""
from(bucket: "${{result_bucket}}")
  {RANGE}
  |> filter(fn: (r) => r._measurement == "tsinfer_forecast" and r._field == "co2" and r.space == "${{space}}")
  |> filter(fn: (r) => r.horizon == "30" or r.horizon == "60")
  |> group(columns: ["horizon"])
  |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
  |> map(fn: (r) => ({{r with series: r.horizon + "분 전 예측"}}))
  |> group(columns: ["series"])
""", "B"),
    ]
    panels.append(timeseries("${space}: 실제 CO₂ vs 30·60분 전에 한 예측", fc_vs_actual, place(8, 9, 0), "suffix: ppm",
                             "예측점은 예측이 가리키는 시각(만든 시각 + 30·60분)에 찍었다. 실제선과 겹칠수록 정확하다.",
                             display="${__field.labels.series}", steps=CO2_STEPS, repeat="space"))
    advance(9)

    err = "|> map(fn: (r) => ({_time: r._time, _start: r._start_a, _stop: r._stop_a, space: r.space, _value: math.abs(x: r._value_a - r._value_b)}))"
    mae_targets = [
        target(f"""import "math"
{actual_co2()}
pred = from(bucket: "${{result_bucket}}")
  {RANGE}
  |> filter(fn: (r) => r._measurement == "tsinfer_forecast" and r._field == "co2" and r.horizon == "60" and contains(value: r.space, set: ${{space:json}}))
  |> group(columns: ["space"])
  |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
join(tables: {{a: actual, b: pred}}, on: ["_time", "space"])
  {err}
  |> set(key: "method", value: "서비스 모델")
  |> group(columns: ["space", "method"])
""", "A"),
        target(f"""import "math"
{actual_co2()}
past = actual |> timeShift(duration: 60m)
join(tables: {{a: actual, b: past}}, on: ["_time", "space"])
  {err}
  |> set(key: "method", value: "현재값 유지")
  |> group(columns: ["space", "method"])
""", "B"),
    ]
    panels.append(timeseries("60분 CO₂ 예측 오차 |실제−예측| — 범례 mean = 선택 구간 MAE", mae_targets, place(24, 9, 0),
                             "suffix: ppm",
                             "과제 3 과 같은 비교를 실시간으로 한다: 서비스 모델 vs 60분 전 값을 그대로 쓴 '현재값 유지'. 5분 평균끼리 비교.",
                             display="${__field.labels.space} · ${__field.labels.method}", calcs=["mean", "max"],
                             extra_overrides=[{"matcher": {"id": "byRegexp", "options": ".*현재값 유지"},
                                               "properties": [{"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [6, 4]}}]}]))
    advance(9)

    panels += [
        stat("60분 뒤 CO₂ 예측", [target(result_field("fc_co2_60"))], place(8, 5, 0), "suffix: ppm", CO2_STEPS,
             "서비스가 마지막으로 낸 60분 예측 (fc_co2_60)", 0),
        stat("1,000ppm 도달까지 (분)", [target(result_field("co2_ventilate_in_minutes"))], place(8, 5, 8), "m",
             thresholds((None, "red"), (30, "orange"), (45, "green")), "예측이 1,000ppm 에 닿지 않으면 값이 없다 (null).", 0),
        stat("마지막 추론 결과 이후 (분)", [target(f"""
from(bucket: "${{result_bucket}}")
  |> range(start: -24h)
  |> filter(fn: (r) => r._measurement == "tsinfer_result" and r._field == "data_ok" and contains(value: r.space, set: ${{space:json}}))
  |> group(columns: ["space"])
  |> last()
  |> map(fn: (r) => ({{r with _value: float(v: int(v: now()) - int(v: r._time)) / 60000000000.0}}))
""")], place(8, 5, 16), "m", thresholds((None, "green"), (3, "orange"), (10, "red")),
             "서비스는 매분 결과를 낸다. 3분을 넘으면 서비스나 브리지가 멈춘 것.", 1),
    ]
    for p in panels[-3:]:
        p["fieldConfig"]["defaults"]["displayName"] = "${__field.labels.space}"
    advance(5)

    panels += [
        state_timeline("환기 경보 (30분 안에 1,000ppm)", "ventilation_alert", place(8, 7, 0), onoff("정상", "환기 필요")),
        state_timeline("이상 경보 (severity=high)", "is_anomaly", place(8, 7, 8), onoff("정상", "이상")),
        state_timeline("데이터 정상 (data_ok)", "data_ok", place(8, 7, 16), onoff("끊김", "정상", off="red", on="green")),
    ]
    advance(7)
    panels.append(state_timeline("사용 여부 추정 (occupied)", "occupied", place(12, 6, 0),
                                 onoff("빈 방", "사용 중", off="#8AB8FF", on="purple"), "데이터가 부족하면 값이 없다 (null)."))
    panels.append({
        "type": "table", "id": nid(), "title": "최근 경보", "datasource": DS, "gridPos": place(12, 6, 12),
        "targets": [target(f"""
from(bucket: "${{result_bucket}}")
  {RANGE}
  |> filter(fn: (r) => r._measurement == "tsinfer_alert" and r._field == "detail" and contains(value: r.space, set: ${{space:json}}))
  |> group()
  |> keep(columns: ["_time", "space", "type", "metric", "severity", "_value"])
  |> rename(columns: {{_value: "detail"}})
  |> sort(columns: ["_time"], desc: true)
  |> limit(n: 100)
""")],
        "fieldConfig": {"defaults": {}, "overrides": [
            {"matcher": {"id": "byName", "options": "severity"}, "properties": [
                {"id": "mappings", "value": [{"type": "value", "options": {
                    "high": {"color": "red", "index": 0}, "medium": {"color": "orange", "index": 1}}}]},
                {"id": "custom.cellOptions", "value": {"type": "color-text"}}]}]},
        "options": {"showHeader": True, "cellHeight": "sm"}})
    advance(6)

    return {
        "__inputs": [],
        "title": "tsinfer 실시간 — 교육장 공기질과 추론 결과",
        "uid": "tsinfer-live",
        "description": "AIP 3.2 시계열 추론 과제. 센서 원본(iot-bucket)과 서비스 추론 결과(tsinfer)를 한 화면에서 본다.",
        "tags": ["tsinfer", "jetson", "aip-3.2"],
        "timezone": "Asia/Seoul",
        "editable": True,
        "graphTooltip": 1,
        "refresh": "1m",
        "schemaVersion": 39,
        "version": 1,
        "time": {"from": "now-12h", "to": "now"},
        "timepicker": {"refresh_intervals": ["30s", "1m", "5m", "15m"]},
        "links": [{"title": "저장소", "type": "link", "url": "https://github.com/AIP-1-3/jenson_AIGJ_01_002",
                   "targetBlank": True}],
        "templating": {"list": [
            {"name": "ds", "label": "InfluxDB", "type": "datasource", "query": "influxdb", "current": {},
             "hide": 0, "refresh": 1, "regex": ""},
            {"name": "bucket", "label": "센서 버킷", "type": "textbox", "query": "iot-bucket",
             "current": {"text": "iot-bucket", "value": "iot-bucket"}, "hide": 0},
            {"name": "result_bucket", "label": "결과 버킷", "type": "textbox", "query": "tsinfer",
             "current": {"text": "tsinfer", "value": "tsinfer"}, "hide": 0},
            {"name": "space", "label": "공간", "type": "query", "datasource": DS,
             "definition": "schema.tagValues location",
             "query": 'import "influxdata/influxdb/schema"\n'
                      'schema.tagValues(bucket: "${bucket}", tag: "location", '
                      'predicate: (r) => r._measurement == "co2" or r._measurement == "temperature", start: -7d)',
             "multi": True, "includeAll": True, "refresh": 1, "sort": 1,
             "current": {"text": ["All"], "value": ["$__all"]}, "hide": 0},
        ]},
        "annotations": {"list": [{"builtIn": 1, "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                                  "enable": True, "hide": True, "name": "Annotations & Alerts", "type": "dashboard"}]},
        "panels": panels,
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)
