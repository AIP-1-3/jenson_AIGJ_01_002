"""SQLite(WAL) 저장소.

- WAL 모드: 수집기가 쓰는 동안 학습·추론이 읽어도 서로 막지 않는다 (과제 1 '생각해 볼 점').
- (space, metric, sensor_ts) 유일 키로 재전송된 같은 메시지를 한 번만 저장한다 (과제 1 심화).
- 센서 시각과 수신 시각을 둘 다 남겨 지연 도착을 나중에 분석할 수 있게 한다.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from .mapping import Reading

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    space       TEXT NOT NULL,
    metric      TEXT NOT NULL,
    ts          TEXT NOT NULL,      -- 분석용 시각: 센서 시각, 없으면 수신 시각 (UTC ISO8601)
    value       REAL NOT NULL,
    received_ts TEXT NOT NULL,
    has_sensor_ts INTEGER NOT NULL,
    source      TEXT NOT NULL,
    PRIMARY KEY (space, metric, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_readings_received ON readings(received_ts);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def write(self, readings: Iterable[Reading]) -> int:
        rows = [
            (
                r.space,
                r.metric,
                (r.sensor_ts or r.received_ts).isoformat(),
                r.value,
                r.received_ts.isoformat(),
                int(r.sensor_ts is not None),
                r.source,
            )
            for r in readings
        ]
        if not rows:
            return 0
        with self.conn:
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO readings VALUES (?, ?, ?, ?, ?, ?, ?)", rows
            )
        return cur.rowcount

    def close(self) -> None:
        self.conn.close()
