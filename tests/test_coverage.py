from datetime import datetime, timedelta, timezone

import pandas as pd

from tsinfer.collector.coverage import coverage, load
from tsinfer.collector.mapping import Reading
from tsinfer.collector.store import Store

T0 = datetime(2026, 10, 21, 11, 0, tzinfo=timezone.utc)


def _reading(space, minute):
    t = T0 + timedelta(minutes=minute)
    return Reading(space=space, device="d1", metric="co2", value=500.0, sensor_ts=t, received_ts=t, source="test")


def test_outage_shows_as_gap(tmp_path):
    # 2시간 동안 1분마다 받다가 30~40분 사이 10분 중단
    db = tmp_path / "t.db"
    s = Store(db)
    s.write([_reading("lab", m) for m in range(120) if not 30 < m < 41])
    s.close()
    start, end = pd.Timestamp(T0), pd.Timestamp(T0 + timedelta(hours=2))
    out = coverage(load(db, T0), start, end, bin_min=10, min_gap=5)
    assert out["spaces"]["lab"]["longest_gap_min"] == 11.0
    assert out["gaps_all"][0]["minutes"] == 11.0
    assert out["spaces"]["lab"]["coverage"] == 1.0  # 10분 구간마다 한 건은 있음
