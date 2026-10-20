"""configs/sensors.yaml 의 매핑으로 MQTT 메시지를 (공간, 항목, 시각, 값) 레코드로 바꾼다.

새 센서·형식 변경은 sensors.yaml 에 source 를 추가해 대응한다 (과제 1-2).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class Reading:
    space: str
    metric: str
    value: float
    sensor_ts: datetime | None   # 센서가 찍은 시각 (없을 수 있음)
    received_ts: datetime        # 수집기가 받은 시각
    source: str


class Source:
    def __init__(self, spec: dict[str, Any]):
        self.name: str = spec["name"]
        self.topic_pattern: str = spec["topic"]
        self.payload_map: dict[str, str] = spec["payload"]
        ts = spec.get("timestamp") or {}
        self.ts_field: str | None = ts.get("field")
        self.ts_format: str = ts.get("format", "iso8601")
        self._regex = self._compile(self.topic_pattern)

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
        if raw is None:
            return None
        if self.ts_format == "epoch_ms":
            return datetime.fromtimestamp(float(raw) / 1000, tz=timezone.utc)
        if self.ts_format == "epoch_s":
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


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
            groups = src.match(topic)
            if groups is None:
                continue
            try:
                body = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                return []
            space = groups.get("space") or body.get("space")
            if not space:
                return []
            try:
                sensor_ts = src.parse_ts(body.get(src.ts_field)) if src.ts_field else None
            except (ValueError, TypeError, OverflowError):
                sensor_ts = None
            return list(self._readings(src, space, body, sensor_ts, received_ts))
        return []

    @staticmethod
    def _readings(src: Source, space: str, body: dict[str, Any], sensor_ts, received_ts) -> Iterable[Reading]:
        for metric, field in src.payload_map.items():
            raw = body.get(field)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            # 0ppm 같은 깨진 값도 일단 원본 그대로 저장한다. 제거는 전처리(과제 2)에서.
            yield Reading(space, metric, value, sensor_ts, received_ts, src.name)
