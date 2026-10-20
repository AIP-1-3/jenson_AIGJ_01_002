"""configs/sensors.yaml 의 매핑으로 MQTT 메시지를 (공간, 항목, 시각, 값) 레코드로 바꾼다.

새 센서·형식 변경은 sensors.yaml 에 source 를 추가해 대응한다 (과제 1-2).

source 설정 요약:
  topic:    구독 패턴. {이름} 자리표시자는 토픽에서 값을 뽑는다 (예: {space}, {dev_eui}).
  each:     (선택) 이 경로의 dict 를 항목별로 돈다. 키가 {key} 로 쓰인다.
            예) xflow 의 {"values": {"회의실": {...}}} → each: values
  space:    공간을 어디서 얻을지.
              from: 토픽 자리표시자 이름, "key"(each 의 키), 또는 페이로드 경로(점 표기)
              map:  (선택) 원래 값 → 공간 이름 표. 표에 없으면 unmapped 규칙을 따른다.
              unmapped: keep(원래 값 그대로 저장, 기본) | drop(버림)
  device:   (선택) 센서 장치 구분. from: 토픽 자리표시자 / "key" / 페이로드 경로. 없으면 "-".
            한 공간에 같은 항목 센서가 여러 개일 때 시계열이 섞이지 않게 한다.
  payload:  저장 항목 이름 → 페이로드 경로(점 표기, 예: object.co2)
  timestamp: field(경로) + format(epoch_s | epoch_ms | iso8601). 없으면 수신 시각만 쓴다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

_MISSING = object()


@dataclass(frozen=True)
class Reading:
    space: str
    device: str
    metric: str
    value: float
    sensor_ts: datetime | None   # 센서가 찍은 시각 (없을 수 있음)
    received_ts: datetime        # 수집기가 받은 시각
    source: str


def get_path(obj: Any, path: str | None) -> Any:
    """'a.b.c' 경로로 중첩 dict 를 따라간다. 없으면 _MISSING."""
    if not path:
        return obj
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return _MISSING
        obj = obj[part]
    return obj


class Source:
    def __init__(self, spec: dict[str, Any]):
        self.name: str = spec["name"]
        self.topic_pattern: str = spec["topic"]
        self.each: str | None = spec.get("each")
        self.payload_map: dict[str, str] = spec["payload"]

        space = spec.get("space") or {"from": "space"}
        self.space_from: str = space["from"]
        self.device_from: str | None = (spec.get("device") or {}).get("from")
        self.space_map: dict[str, str] = {str(k): str(v) for k, v in (space.get("map") or {}).items()}
        self.space_unmapped: str = space.get("unmapped", "keep")
        if self.space_unmapped not in ("keep", "drop"):
            raise ValueError(f"{self.name}: space.unmapped 는 keep 또는 drop 이어야 합니다.")

        ts = spec.get("timestamp") or {}
        self.ts_field: str | None = ts.get("field")
        self.ts_format: str = ts.get("format", "iso8601")
        self._regex = self._compile(self.topic_pattern)
        self._topic_vars = set(self._regex.groupindex)

    @staticmethod
    def _compile(pattern: str) -> re.Pattern[str]:
        parts = re.split(r"(\{\w+\})", pattern)
        out = []
        for p in parts:
            m = re.fullmatch(r"\{(\w+)\}", p)
            out.append(f"(?P<{m.group(1)}>[^/]+)" if m else re.escape(p))
        return re.compile("^" + "".join(out) + "$")

    @property
    def subscription(self) -> str:
        """MQTT 구독용 토픽 ({x} → +)."""
        return re.sub(r"\{\w+\}", "+", self.topic_pattern)

    def match(self, topic: str) -> dict[str, str] | None:
        m = self._regex.match(topic)
        return m.groupdict() if m else None

    def parse_ts(self, raw: Any) -> datetime | None:
        if raw is None or raw is _MISSING:
            return None
        if self.ts_format == "epoch_ms":
            return datetime.fromtimestamp(float(raw) / 1000, tz=timezone.utc)
        if self.ts_format == "epoch_s":
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    def _lookup(self, source: str, topic_vars: dict[str, str], key: str | None, body: Any) -> str | None:
        if source in self._topic_vars:
            raw: Any = topic_vars.get(source)
        elif source == "key":
            raw = key
        else:
            raw = get_path(body, source)
        if raw is None or raw is _MISSING or raw == "":
            return None
        return str(raw)

    def resolve_device(self, topic_vars: dict[str, str], key: str | None, body: Any) -> str:
        if not self.device_from:
            return "-"
        return self._lookup(self.device_from, topic_vars, key, body) or "-"

    def resolve_space(self, topic_vars: dict[str, str], key: str | None, body: Any) -> str | None:
        raw = self._lookup(self.space_from, topic_vars, key, body)
        if raw is None:
            return None
        if raw in self.space_map:
            return self.space_map[raw]
        return raw if self.space_unmapped == "keep" else None


class Mapper:
    def __init__(self, config: dict[str, Any]):
        self.sources = [Source(s) for s in config.get("sources", [])]
        if not self.sources:
            raise ValueError("sensors.yaml 에 sources 가 없습니다.")

    @property
    def subscriptions(self) -> list[str]:
        return sorted({s.subscription for s in self.sources})

    def parse(self, topic: str, payload: bytes, received_ts: datetime) -> list[Reading]:
        """매칭되는 source 가 없거나 깨진 메시지는 빈 리스트 (수집기는 죽지 않는다)."""
        for src in self.sources:
            topic_vars = src.match(topic)
            if topic_vars is None:
                continue
            try:
                body = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                return []
            if not isinstance(body, dict):
                return []
            try:
                sensor_ts = src.parse_ts(get_path(body, src.ts_field)) if src.ts_field else None
            except (ValueError, TypeError, OverflowError):
                sensor_ts = None
            return list(self._records(src, topic_vars, body, sensor_ts, received_ts))
        return []

    @staticmethod
    def _records(src: Source, topic_vars, body: dict[str, Any], sensor_ts, received_ts) -> Iterable[Reading]:
        if src.each:
            group = get_path(body, src.each)
            items = group.items() if isinstance(group, dict) else []
        else:
            items = [(None, body)]
        for key, node in items:
            space = src.resolve_space(topic_vars, key, body if key is None else node)
            if space is None and src.space_from not in src._topic_vars and key is not None:
                # each 안에서 경로를 못 찾으면 메시지 전체에서 한 번 더 찾는다
                space = src.resolve_space(topic_vars, key, body)
            if not space:
                continue
            device = src.resolve_device(topic_vars, key, body)
            for metric, path in src.payload_map.items():
                raw = get_path(node, path)
                if raw is _MISSING or raw is None:
                    continue
                try:
                    value = float(raw)  # bool 은 1.0/0.0 으로 저장
                except (TypeError, ValueError):
                    continue
                # 0ppm 같은 깨진 값도 일단 원본 그대로 저장한다. 제거는 전처리(과제 2)에서.
                yield Reading(space, device, metric, value, sensor_ts, received_ts, src.name)
