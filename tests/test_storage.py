import sqlite3
from datetime import datetime, timedelta, timezone

from tsinfer.collector.mapping import Reading
from tsinfer.collector.storage import POLICIES, analyze, apply, connect_ro, plan, project, report
from tsinfer.collector.store import Store

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _fill(store: Store, days: int = 40, step_min: int = 1, series=(("lab", "d1"), ("office", "d2"))):
    """days 일 전부터 지금까지 step_min 간격 co2. 수신 시각 = 센서 시각 + 2초."""
    t = NOW - timedelta(days=days)
    batch = []
    while t < NOW:
        for sp, dev in series:
            batch.append(Reading(sp, dev, "co2", 500.0 + t.minute, t, t + timedelta(seconds=2), "test"))
        t += timedelta(minutes=step_min)
    store.write(batch)
    return len(batch)


def test_store_counts_ignored_duplicates(tmp_path):
    store = Store(tmp_path / "t.db", dup_stats=True)
    r = [Reading("lab", "d1", "co2", 600.0, NOW, NOW, "test")]
    assert store.write(r) == 1 and store.write(r) == 0          # 반환값 의미는 그대로
    assert (store.received, store.inserted, store.duplicates_ignored) == (2, 1, 1)
    row = store.conn.execute("SELECT received, inserted FROM ingest_stats").fetchone()
    assert row == (2, 1)


def test_store_default_has_no_stats_table(tmp_path, monkeypatch):
    monkeypatch.delenv("STORE_DUP_STATS", raising=False)
    store = Store(tmp_path / "t.db")
    store.write([Reading("lab", "d1", "co2", 600.0, NOW, NOW, "test")])
    names = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master")}
    assert "ingest_stats" not in names and store.duplicates_ignored == 0


def test_analyze_projects_and_plans(tmp_path):
    db = tmp_path / "t.db"
    store = Store(db)
    n = _fill(store, days=10, step_min=5)
    # 센서 시각 없는 재전송 의심: 같은 값이 1초 뒤 다시
    store.write([Reading("lab", "d9", "co2", 700.0, None, NOW - timedelta(hours=1), "t"),
                 Reading("lab", "d9", "co2", 700.0, None, NOW - timedelta(hours=1) + timedelta(seconds=1), "t")])
    store.close()
    a = analyze(db, now=NOW)
    assert a["rows"] == n + 2 and a["bytes_per_row"] > 0
    assert abs(a["rows_per_day"] - 2 * 288) < 5
    assert a["dups"]["suspect_no_ts"] == 1 and a["dups"]["late"][300] == 0
    proj = project(a)
    keep = proj.set_index("policy").loc["keep_all", "센서 ×1 (MB)"]
    assert proj.set_index("policy").loc["raw30_5m", "센서 ×1 (MB)"] < keep
    assert proj.set_index("policy").loc["keep_all", "센서 ×5 (MB)"] > keep
    with connect_ro(db) as conn:
        pl = plan(conn, POLICIES["raw30_5m"], NOW)
    assert pl["raw_rows_to_delete"] == 0                         # 10일치뿐 → 지울 것 없음
    assert "## 해석 (직접 작성)" in report(a, proj, pl)


def test_apply_dry_then_delete_keeps_aggregates(tmp_path):
    db = tmp_path / "t.db"
    store = Store(db)
    _fill(store, days=40, step_min=1)
    store.close()
    conn = sqlite3.connect(db)
    before = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    old_cut = (NOW.replace(hour=0) - timedelta(days=30)).isoformat()
    old_n = conn.execute("SELECT COUNT(*) FROM readings WHERE ts < ?", (old_cut,)).fetchone()[0]
    conn.close()

    p = POLICIES["raw30_5m180_1h"]
    st = apply(db, p, now=NOW, delete=False, log=lambda *_: None)    # 집계만
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == before
    assert st["days_aggregated"] >= 39

    st = apply(db, p, now=NOW, delete=True, log=lambda *_: None)
    assert st["raw_deleted"] == old_n and st["days_skipped"] == 0
    # 지운 기간도 집계 건수 합으로 원본 행 수가 보존된다
    n5 = conn.execute("SELECT SUM(n) FROM readings_5min WHERE ts < ?", (old_cut,)).fetchone()[0]
    n1 = conn.execute("SELECT SUM(n) FROM readings_1h WHERE ts < ?", (old_cut,)).fetchone()[0]
    assert n5 == n1 == old_n
    m = conn.execute("SELECT mean, min, max, n FROM readings_5min WHERE space='lab' ORDER BY ts LIMIT 1").fetchone()
    assert m[3] == 5 and m[1] <= m[0] <= m[2]


def test_policy_without_aggregates_never_deletes(tmp_path):
    db = tmp_path / "t.db"
    store = Store(db)
    n = _fill(store, days=3, step_min=10)
    store.close()
    st = apply(db, POLICIES["keep_all"], now=NOW, delete=True, log=lambda *_: None)
    assert st["raw_deleted"] == 0
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM readings").fetchone()[0] == n
