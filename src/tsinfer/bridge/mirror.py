"""수집기 SQLite 센서값 → (Jetson 안의) InfluxDB 복사. 학교·교육장 InfluxDB 에 닿지 않을 때만 쓴다.

Jetson 에서 원격 InfluxDB 포트가 막혀 있으면 Grafana 가 센서 원본(iot-bucket)을 읽을 수 없다.
수집기는 이미 같은 센서를 MQTT 로 받아 SQLite 에 쌓고 있으므로, 그 값을 로컬 InfluxDB 에
iot-bucket 과 같은 구조(measurement=항목, field value, 태그 location·device_id)로 옮기면
대시보드 쿼리를 하나도 바꾸지 않고 그대로 쓸 수 있다.

- 수집기 DB 는 읽기 전용으로 연다 (수집기와 부딪히지 않게).
- received_ts 기준으로 새로 들어온 행만 옮긴다. 시작할 때는 최근 backfill_hours 시간을 먼저 옮긴다.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger("bridge.mirror")
BATCH = 20_000


def _esc_key(s: str) -> str:
    """line protocol 의 measurement·태그 이름/값 escape (쉼표·등호·공백)."""
    return s.replace("\\", "\\\\").replace(",", "\\,").replace("=", "\\=").replace(" ", "\\ ")


def row_to_line(ts: str, space: str, device: str, metric: str, value: float) -> str:
    t = int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    return (f"{_esc_key(metric)},device_id={_esc_key(device or '-')},location={_esc_key(space or '-')} "
            f"value={float(value)!r} {t}")


def read_since(db_path: Path, cursor: str, limit: int = BATCH) -> list[tuple]:
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        return conn.execute(
            "SELECT ts, space, device, metric, value, received_ts FROM readings "
            "WHERE received_ts > ? ORDER BY received_ts LIMIT ?", (cursor, limit)).fetchall()


class SensorMirror:
    def __init__(self, db_path: Path, sink, backfill_hours: float = 24, interval_s: float = 60):
        self.db_path, self.sink, self.interval_s = db_path, sink, interval_s
        self.cursor = (datetime.now(timezone.utc) - timedelta(hours=backfill_hours)).isoformat()
        self._stop = threading.Event()

    def step(self) -> int:
        """새 행을 옮기고 옮긴 행 수를 돌려준다. 한 번에 BATCH 행씩, 밀린 만큼 반복."""
        total = 0
        while True:
            rows = read_since(self.db_path, self.cursor)
            if not rows:
                return total
            self.sink.write([row_to_line(*r[:5]) for r in rows])
            self.cursor = rows[-1][5]
            total += len(rows)
            if len(rows) < BATCH:
                return total

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                n = self.step()
                if n:
                    log.info("센서 %d행 복사 (~%s)", n, self.cursor[:19])
            except Exception:  # DB 잠김 등 일시 오류로 브리지가 죽지 않게
                log.exception("센서 복사 실패 — 다음 주기에 다시")
            self._stop.wait(self.interval_s)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="sensor-mirror", daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()
