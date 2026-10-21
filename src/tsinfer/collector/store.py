"""SQLite(WAL) 저장소.

- WAL 모드: 수집기가 쓰는 동안 학습·추론이 읽어도 서로 막지 않는다 (과제 1 '생각해 볼 점').
- (space, device, metric, ts) 유일 키로 재전송된 같은 메시지를 한 번만 저장한다 (과제 1 심화).
- 센서 시각과 수신 시각을 둘 다 남겨 지연 도착을 나중에 분석할 수 있게 한다.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .mapping import Reading

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    space       TEXT NOT NULL,
    device      TEXT NOT NULL,      -- 같은 공간의 여러 센서를 구분 (없으면 '-')
    metric      TEXT NOT NULL,
    ts          TEXT NOT NULL,      -- 분석용 시각: 센서 시각, 없으면 수신 시각 (UTC ISO8601)
    value       REAL NOT NULL,
    received_ts TEXT NOT NULL,
    has_sensor_ts INTEGER NOT NULL,
    source      TEXT NOT NULL,
    PRIMARY KEY (space, device, metric, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_readings_received ON readings(received_ts);
"""

# 과제 1 심화: 재전송(중복) 건수 기록. 환경변수 STORE_DUP_STATS=1 일 때만 표를 만들고 쓴다.
# 수집기(__main__)를 고치지 않고 켤 수 있게 Store 안에서 처리한다 (켜려면 수집기 재시작 필요).
STATS_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_stats (
    day       TEXT PRIMARY KEY,     -- 수신 날짜 (UTC, YYYY-MM-DD)
    received  INTEGER NOT NULL,     -- write 가 받은 행 수
    inserted  INTEGER NOT NULL      -- 실제 저장된 행 수 (차이 = 같은 키라 무시된 재전송)
);
"""


class Store:
    def __init__(self, path: Path, dup_stats: bool | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        # 메모리 누적 카운터는 항상 (비용 없음). 표 기록은 선택.
        self.received = 0
        self.inserted = 0
        self.dup_stats = os.getenv("STORE_DUP_STATS") == "1" if dup_stats is None else dup_stats
        if self.dup_stats:
            self.conn.executescript(STATS_SCHEMA)

    @property
    def duplicates_ignored(self) -> int:
        """이 Store 를 연 뒤 같은 키라서 버린 행 수 (재전송 추정치)."""
        return self.received - self.inserted

    def write(self, readings: Iterable[Reading]) -> int:
        rows = [
            (
                r.space,
                r.device,
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
                "INSERT OR IGNORE INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            n = cur.rowcount
            if self.dup_stats:   # 같은 트랜잭션: 저장과 집계가 어긋나지 않는다
                self.conn.execute(
                    "INSERT INTO ingest_stats VALUES (?, ?, ?) ON CONFLICT(day) DO UPDATE SET "
                    "received = received + excluded.received, inserted = inserted + excluded.inserted",
                    (datetime.now(timezone.utc).strftime("%Y-%m-%d"), len(rows), n),
                )
        self.received += len(rows)
        self.inserted += n
        return n   # 의미 그대로: 이번에 실제 저장된 행 수

    def close(self) -> None:
        self.conn.close()
