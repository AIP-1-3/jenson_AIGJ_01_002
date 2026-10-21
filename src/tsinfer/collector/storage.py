"""과제 1 심화: 저장소 크기 추정·보관 정책, 재전송(중복) 찾기.

    python -m tsinfer.collector.storage                          # 분석 + 정책 미리보기 → reports/01b_storage.md
    python -m tsinfer.collector.storage --policy raw90_5m --apply           # 집계 표만 채움 (삭제 없음)
    python -m tsinfer.collector.storage --policy raw90_5m --apply --delete  # 집계 확인 후 오래된 원본 삭제

설계
- 분석은 읽기 전용(file:...?mode=ro)으로 연다. 수집기가 쓰는 중이어도 막지 않는다.
- 1년 뒤 크기 = 지금 쓰는 크기 + 최근 7일 하루 평균 행 수 × 365 × 행당 바이트(인덱스 포함 실측).
- 보관 정책: 원본은 N일만, 그보다 오래된 구간은 5분·1시간 집계(평균·최소·최대·건수)로 남긴다.
  학습 입력이 5분 간격(preprocess.to_wide)이라 5분 집계만 있어도 재학습할 수 있다.
- 데이터 손실 방지: 기본은 dry-run(지울 건수만 출력). 실제 삭제는 --apply --delete 둘 다 있어야 하고,
  하루 단위 한 트랜잭션 안에서 '집계를 다시 써 넣고 → 집계 건수 합 = 원본 행 수 확인 → 삭제' 순서로만 지운다.
  확인이 어긋나면 그날은 지우지 않는다. 집계 표가 없는 정책은 삭제 자체를 거부한다.
- 삭제해도 SQLite 파일은 줄지 않고 빈 페이지를 재사용한다 (VACUUM 은 수집기를 멈춰야 해서 하지 않는다).
"""
from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsinfer.config import ROOT, db_path

DAY = timedelta(days=1)
AGG_ROW_RATIO = 0.6       # 집계 행 크기 / 원본 행 크기 (가정: received_ts·source·수신 인덱스가 없다). 실측 가능하면 덮어씀
SENSOR_SCENARIOS = (1, 2, 5)   # 센서(시계열) 수가 지금의 몇 배가 되는지
DUP_WINDOW_S = 5          # 센서 시각 없는 행: 같은 값이 이 초 안에 다시 오면 재전송 의심
LATE_S = (300, 3600)      # 센서 시각 있는 행: 수신이 이만큼 늦으면 늦은 도착(재전송·버퍼 방출 의심)


@dataclass(frozen=True)
class Policy:
    """보관 기간(일). None = 무기한, 0 = 만들지 않음."""

    name: str
    desc: str
    raw_days: int | None
    agg5_days: int | None
    agg1h_days: int | None


POLICIES = {p.name: p for p in (
    Policy("keep_all", "원본 무기한 (지금)", None, 0, 0),
    Policy("raw90_5m", "원본 90일 + 5분 집계 무기한", 90, None, 0),
    Policy("raw30_5m", "원본 30일 + 5분 집계 무기한", 30, None, 0),
    Policy("raw30_5m180_1h", "원본 30일 + 5분 집계 180일 + 1시간 집계 무기한", 30, 180, None),
)}

AGG_TABLES = {"readings_5min": 300, "readings_1h": 3600}
AGG_SCHEMA = """
CREATE TABLE IF NOT EXISTS {t} (
    space  TEXT NOT NULL,
    device TEXT NOT NULL,
    metric TEXT NOT NULL,
    ts     TEXT NOT NULL,       -- 구간 시작 (UTC ISO8601)
    mean   REAL NOT NULL,
    min    REAL NOT NULL,
    max    REAL NOT NULL,
    n      INTEGER NOT NULL,    -- 원본 행 수 (삭제 전 확인에 쓴다)
    PRIMARY KEY (space, device, metric, ts)
) WITHOUT ROWID;
"""


def connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)


def _one(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Any:
    return conn.execute(sql, params).fetchone()[0]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return _one(conn, "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)) > 0


def _table_bytes(conn: sqlite3.Connection, name: str) -> int | None:
    """dbstat 가상 표가 있으면 표+인덱스 실제 바이트. 없으면 None."""
    try:
        idx = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE tbl_name=?", (name,))]
        return int(sum(_one(conn, "SELECT COALESCE(SUM(pgsize),0) FROM dbstat WHERE name=?", (n,)) for n in idx))
    except sqlite3.Error:
        return None


# ---------------------------------------------------------------- 분석

def analyze(path: Path, now: datetime | None = None, days: int = 7) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    files = {suf or "db": (Path(f"{path}{suf}").stat().st_size if Path(f"{path}{suf}").exists() else 0)
             for suf in ("", "-wal", "-shm")}
    conn = connect_ro(path)
    try:
        page = _one(conn, "PRAGMA page_size")
        used = (_one(conn, "PRAGMA page_count") - _one(conn, "PRAGMA freelist_count")) * page
        rows = _one(conn, "SELECT COUNT(*) FROM readings")
        first_rx, last_rx = conn.execute("SELECT MIN(received_ts), MAX(received_ts) FROM readings").fetchone()
        ts_min, ts_max = conn.execute("SELECT MIN(ts), MAX(ts) FROM readings").fetchone()
        non_utc = _one(conn, "SELECT COUNT(*) FROM readings WHERE ts NOT LIKE '%+00:00'")
        raw_bytes = _table_bytes(conn, "readings") or used
        start = now - timedelta(days=days)
        if first_rx:
            start = max(start, pd.Timestamp(first_rx).to_pydatetime())
        window_days = max((now - start).total_seconds() / 86400, 1e-6)
        daily = pd.read_sql_query(
            "SELECT substr(received_ts, 1, 10) AS day, COUNT(*) AS rows FROM readings "
            "WHERE received_ts >= ? GROUP BY day ORDER BY day", conn, params=(start.isoformat(),))
        series = pd.read_sql_query(
            "SELECT space, device, metric, COUNT(*) AS rows FROM readings WHERE received_ts >= ? "
            "GROUP BY space, device, metric ORDER BY space, device, metric", conn, params=(start.isoformat(),))
        agg_bytes = {t: (_table_bytes(conn, t), _one(conn, f"SELECT COUNT(*) FROM {t}"))
                     for t in AGG_TABLES if _has_table(conn, t)}
        dups = duplicates(conn, start)
    finally:
        conn.close()
    series["rows_per_day"] = series["rows"] / window_days
    bpr = raw_bytes / rows if rows else 0.0
    agg_bpr = bpr * AGG_ROW_RATIO
    measured = [(b / n) for b, n in agg_bytes.values() if b and n]
    if measured:
        agg_bpr = sum(measured) / len(measured)
    span_days = 0.0
    if ts_min and ts_max:
        span_days = (pd.Timestamp(ts_max) - pd.Timestamp(ts_min)).total_seconds() / 86400
    return {
        "now": now, "path": path, "files": files, "used_bytes": used, "rows": rows,
        "first_received": first_rx, "last_received": last_rx, "ts_min": ts_min, "ts_max": ts_max,
        "span_days": span_days, "non_utc_rows": non_utc, "window_start": start, "window_days": window_days,
        "daily": daily, "series": series, "rows_per_day": float(series["rows"].sum() / window_days),
        "bytes_per_row": bpr, "agg_bytes_per_row": agg_bpr, "agg_measured": bool(measured),
        "dups": dups,
    }


def project(a: dict[str, Any], horizon_days: int = 365) -> pd.DataFrame:
    """정책 × 센서 시나리오별 horizon_days 뒤 크기 (MB). 지금까지 쌓인 기간 + horizon 을 운영 기간으로 본다."""
    T = a["span_days"] + horizon_days
    per = a["series"]["rows_per_day"]
    agg5_day = float(per.clip(upper=86400 / 300).sum())    # 시계열 하나가 5분에 1행 이하로 줄어든다
    agg1h_day = float(per.clip(upper=24).sum())
    rows = []
    for p in POLICIES.values():
        row = {"policy": p.name, "설명": p.desc}
        for k in SENSOR_SCENARIOS:
            def kept(d):   # 보관 일수
                return 0.0 if d == 0 else (T if d is None else min(T, d))
            b = (kept(p.raw_days) * a["rows_per_day"] * a["bytes_per_row"]
                 + (kept(p.agg5_days) * agg5_day + kept(p.agg1h_days) * agg1h_day) * a["agg_bytes_per_row"]) * k
            row[f"센서 ×{k} (MB)"] = round(b / 1e6, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def duplicates(conn: sqlite3.Connection, since: datetime) -> dict[str, Any]:
    """수집기 로그를 바꾸지 않고 DB 흔적으로 재전송을 찾는다.

    - 센서 시각이 있는 행: 같은 키(공간·장치·항목·센서 시각)로 다시 오면 INSERT OR IGNORE 로 버려져 흔적이 없다.
      대신 '늦게 도착한' 행(수신 − 센서 시각이 큼)은 재전송·게이트웨이 버퍼 방출의 흔적이다.
    - 센서 시각이 없는 행: 수신 시각이 키라 재전송이 걸러지지 않는다. 같은 값이 몇 초 안에 다시 온 행을 센다.
    """
    s = since.isoformat()
    out: dict[str, Any] = {}
    out["with_sensor_ts"], out["without_sensor_ts"] = conn.execute(
        "SELECT SUM(has_sensor_ts=1), SUM(has_sensor_ts=0) FROM readings WHERE received_ts >= ?", (s,)).fetchone()
    lat = "(julianday(received_ts) - julianday(ts)) * 86400"
    out["late"] = {sec: _one(conn, f"SELECT COUNT(*) FROM readings WHERE received_ts >= ? AND has_sensor_ts=1 "
                                   f"AND {lat} > ?", (s, sec)) for sec in LATE_S}
    out["max_delay_s"] = _one(conn, f"SELECT MAX({lat}) FROM readings WHERE received_ts >= ? AND has_sensor_ts=1", (s,))
    out["suspect_no_ts"] = _one(conn, f"""
        SELECT COUNT(*) FROM (
          SELECT value, ts, LAG(value) OVER w AS pv, LAG(ts) OVER w AS pts FROM readings
          WHERE received_ts >= ? AND has_sensor_ts=0
          WINDOW w AS (PARTITION BY space, device, metric ORDER BY ts))
        WHERE pv = value AND (julianday(ts) - julianday(pts)) * 86400 <= {DUP_WINDOW_S}""", (s,))
    out["ingest_stats"] = (pd.read_sql_query("SELECT day, received, inserted, received - inserted AS duplicates "
                                             "FROM ingest_stats ORDER BY day", conn)
                           if _has_table(conn, "ingest_stats") else None)
    return out


# ---------------------------------------------------------------- 정책 실행

def _day(ts: str | None) -> datetime | None:
    return pd.Timestamp(ts).tz_convert("UTC").floor("D").to_pydatetime() if ts else None


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).isoformat()


def _bucket(sec: int) -> str:
    return f"strftime('%Y-%m-%dT%H:%M:%S+00:00', (CAST(strftime('%s', ts) AS INTEGER) / {sec}) * {sec}, 'unixepoch')"


def _aggregate_day(conn: sqlite3.Connection, table: str, d: datetime) -> None:
    """원본 하루치 → 집계 표 (다시 돌려도 같은 결과: INSERT OR REPLACE)."""
    conn.execute(
        f"INSERT OR REPLACE INTO {table} SELECT space, device, metric, {_bucket(AGG_TABLES[table])} AS b, "
        f"AVG(value), MIN(value), MAX(value), COUNT(*) FROM readings WHERE ts >= ? AND ts < ? "
        f"GROUP BY space, device, metric, b", (_iso(d), _iso(d + DAY)))


def _count(conn: sqlite3.Connection, table: str, d: datetime) -> int:
    col = "COUNT(*)" if table == "readings" else "COALESCE(SUM(n), 0)"
    return _one(conn, f"SELECT {col} FROM {table} WHERE ts >= ? AND ts < ?", (_iso(d), _iso(d + DAY)))


def plan(conn: sqlite3.Connection, policy: Policy, now: datetime) -> dict[str, Any]:
    """무엇을 지울지 건수만 (dry-run). 아무것도 바꾸지 않는다."""
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    out: dict[str, Any] = {"policy": policy.name, "desc": policy.desc}
    tables = [t for t, d in (("readings_5min", policy.agg5_days), ("readings_1h", policy.agg1h_days)) if d != 0]
    out["agg_tables"] = tables
    if policy.raw_days is not None:
        cut = today - timedelta(days=policy.raw_days)
        out["raw_cutoff"] = _iso(cut)
        out["raw_rows_to_delete"] = _one(conn, "SELECT COUNT(*) FROM readings WHERE ts < ?", (_iso(cut),))
    if policy.agg5_days not in (None, 0) and _has_table(conn, "readings_5min"):
        cut5 = today - timedelta(days=policy.agg5_days)
        out["agg5_cutoff"] = _iso(cut5)
        out["agg5_rows_to_delete"] = _one(conn, "SELECT COUNT(*) FROM readings_5min WHERE ts < ?", (_iso(cut5),))
    return out


def apply(path: Path, policy: Policy, now: datetime | None = None, delete: bool = False, log=print) -> dict[str, int]:
    """집계 표를 채우고, delete=True 면 확인을 통과한 날만 오래된 원본(·5분 집계)을 지운다."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    tables = [t for t, d in (("readings_5min", policy.agg5_days), ("readings_1h", policy.agg1h_days)) if d != 0]
    stats = {"days_aggregated": 0, "raw_deleted": 0, "agg5_deleted": 0, "days_skipped": 0}
    if not tables:
        log("집계 표가 없는 정책이라 할 일이 없습니다 (원본을 지우려면 집계가 먼저 있어야 함).")
        return stats
    conn = sqlite3.connect(path, timeout=30)
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        if _one(conn, "SELECT COUNT(*) FROM readings WHERE ts NOT LIKE '%+00:00'"):
            raise RuntimeError("UTC(+00:00)가 아닌 ts 가 있어 하루 경계 비교를 믿을 수 없습니다. 실행하지 않습니다.")
        for t in tables:
            conn.executescript(AGG_SCHEMA.format(t=t))
        first = _day(_one(conn, "SELECT MIN(ts) FROM readings"))
        if first is None:
            return stats
        # 1) 집계: 마지막으로 집계한 날부터 어제까지 (완성된 날만). 하루 = 한 트랜잭션 (수집기를 오래 막지 않게)
        last = [_day(_one(conn, f"SELECT MAX(ts) FROM {t}")) for t in tables]
        d = min([x for x in last if x] or [first]) if all(last) else first
        while d < today:
            with conn:
                for t in tables:
                    _aggregate_day(conn, t, d)
            stats["days_aggregated"] += 1
            d += DAY
        if not delete:
            return stats
        # 2) 원본 삭제: 집계를 다시 쓰고 건수 합을 맞춘 뒤에만
        if policy.raw_days is not None:
            d, cut = first, today - timedelta(days=policy.raw_days)
            while d < cut:
                with conn:
                    for t in tables:
                        _aggregate_day(conn, t, d)
                    raw_n = _count(conn, "readings", d)
                    if raw_n and all(_count(conn, t, d) == raw_n for t in tables):
                        stats["raw_deleted"] += conn.execute(
                            "DELETE FROM readings WHERE ts >= ? AND ts < ?", (_iso(d), _iso(d + DAY))).rowcount
                    elif raw_n:
                        stats["days_skipped"] += 1
                        log(f"{d:%Y-%m-%d}: 집계 건수가 원본과 달라 지우지 않음")
                d += DAY
        # 3) 5분 집계 보관 기간: 1시간 집계 건수 합이 같을 때만
        if policy.agg5_days not in (None, 0) and "readings_1h" in tables:
            cut5 = today - timedelta(days=policy.agg5_days)
            with conn:
                old = [r[0] for r in conn.execute(
                    "SELECT DISTINCT substr(ts, 1, 10) FROM readings_5min WHERE ts < ?", (_iso(cut5),))]
            for day in old:
                d = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
                with conn:
                    if _count(conn, "readings_5min", d) == _count(conn, "readings_1h", d):
                        stats["agg5_deleted"] += conn.execute(
                            "DELETE FROM readings_5min WHERE ts >= ? AND ts < ?", (_iso(d), _iso(d + DAY))).rowcount
                    else:
                        stats["days_skipped"] += 1
    finally:
        conn.close()
    return stats


# ---------------------------------------------------------------- 보고

def _mb(b: float) -> str:
    return f"{b / 1e6:,.1f}"


def report(a: dict[str, Any], proj: pd.DataFrame, pl: dict[str, Any]) -> str:
    dups = a["dups"]
    files = pd.DataFrame([{"파일": k, "MB": _mb(v)} for k, v in a["files"].items()]
                         + [{"파일": "합계", "MB": _mb(sum(a["files"].values()))}])
    one_year = pd.DataFrame([{
        "센서 수": f"×{k}",
        "하루 행 수": f"{a['rows_per_day'] * k:,.0f}",
        "하루 MB": _mb(a["rows_per_day"] * k * a["bytes_per_row"]),
        "1년 뒤 MB (원본 무기한)": _mb(a["used_bytes"] + a["rows_per_day"] * k * a["bytes_per_row"] * 365),
    } for k in SENSOR_SCENARIOS])
    ser = a["series"].copy()
    ser["rows_per_day"] = ser["rows_per_day"].round(0)
    late = pd.DataFrame([{"구분": f"수신 지연 > {s // 60}분", "행 수": n} for s, n in dups["late"].items()]
                        + [{"구분": "최대 수신 지연(초)", "행 수": round(dups["max_delay_s"] or 0, 1)},
                           {"구분": f"센서 시각 없음 + {DUP_WINDOW_S}초 안 같은 값 (재전송 의심)", "행 수": dups["suspect_no_ts"]}])
    lines = [
        "# 과제 1 심화. 저장소 크기 추정과 보관 정책, 재전송 찾기",
        "",
        "## 측정 조건",
        f"- 측정 시각 {a['now']:%Y-%m-%d %H:%M} UTC, DB `{Path(a['path']).name}` (읽기 전용으로 열어 수집기와 부딪히지 않음)",
        f"- 저장 행 {a['rows']:,}, 센서 시각 범위 {a['ts_min']} ~ {a['ts_max']} ({a['span_days']:.1f}일), "
        f"첫 수신 {a['first_received']}",
        f"- 하루 평균은 최근 {a['window_days']:.1f}일(최대 7일, 수신 시각 기준)로 계산",
        f"- 행당 바이트 {a['bytes_per_row']:.1f} (원본 표 + 수신 시각 인덱스, 사용 중인 페이지 기준). "
        f"집계 행 {a['agg_bytes_per_row']:.1f} 바이트 ({'실측' if a['agg_measured'] else f'가정: 원본의 {AGG_ROW_RATIO:.0%}'})",
        f"- UTC(+00:00)가 아닌 ts 행: {a['non_utc_rows']:,} (0 이어야 정책 실행이 하루 경계를 바르게 자른다)",
        "",
        "## 1. 지금 크기",
        "",
        files.to_markdown(index=False),
        "",
        f"사용 중 페이지 {_mb(a['used_bytes'])} MB. -wal 은 체크포인트 전 변경분이라 시간에 따라 오르내린다.",
        "",
        "## 2. 최근 수신량",
        "",
        a["daily"].to_markdown(index=False),
        "",
        ser.to_markdown(index=False),
        "",
        "## 3. 1년 뒤 크기 (원본을 그대로 쌓을 때)",
        "",
        one_year.to_markdown(index=False),
        "",
        "## 4. 보관 정책별 크기 (지금까지 기간 + 365일 운영, MB)",
        "",
        "집계 = 시계열(공간·장치·항목)마다 5분/1시간 구간의 평균·최소·최대·건수. "
        "기록 주기가 구간보다 길면 집계로 줄지 않는다(시계열당 하루 min(원본, 288 또는 24)행).",
        "",
        proj.to_markdown(index=False),
        "",
        f"## 5. 정책 실행 미리보기 (dry-run: `{pl['policy']}` {pl['desc']})",
        "",
        f"- 집계 표: {', '.join(pl['agg_tables']) or '없음'}",
        f"- 지울 원본 행 (ts < {pl.get('raw_cutoff', '-')}): {pl.get('raw_rows_to_delete', 0):,}",
        f"- 지울 5분 집계 행: {pl.get('agg5_rows_to_delete', 0):,}",
        "- 실행: `--apply`(집계만) → 확인 후 `--apply --delete`. 하루 단위로 집계 건수 합 = 원본 행 수일 때만 지운다.",
        "",
        "## 6. 재전송(중복) 찾기",
        "",
        "- 막는 방법: `(space, device, metric, ts)` 유일 키 + `INSERT OR IGNORE` — 같은 메시지가 다시 와도 한 번만 저장.",
        "- 버려진 건수는 DB 에 흔적이 남지 않는다. 그래서 (1) DB 의 간접 흔적을 세고, "
        "(2) `STORE_DUP_STATS=1` 로 수집기를 다시 띄우면 `ingest_stats` 표에 날짜별 받은 행·저장 행이 쌓인다.",
        f"- 최근 구간 센서 시각 있는 행 {dups['with_sensor_ts'] or 0:,} / 없는 행 {dups['without_sensor_ts'] or 0:,} "
        "(없는 행은 수신 시각이 키라 재전송이 걸러지지 않는다)",
        "",
        late.to_markdown(index=False),
        "",
    ]
    if dups["ingest_stats"] is not None and len(dups["ingest_stats"]):
        lines += ["### 수집기 카운터 (ingest_stats)", "", dups["ingest_stats"].to_markdown(index=False), ""]
    else:
        lines += ["- `ingest_stats` 표 없음: 지금 가동 중인 수집기는 카운터 이전 코드라, "
                  "`.env` 에 `STORE_DUP_STATS=1` 을 넣고 수집기를 재시작해야 버린 중복 수가 쌓인다.", ""]
    lines += ["## 해석 (직접 작성)", "- 1년 뒤 크기와 Jetson NVMe(233GB) 대비 여유:",
              "- 고른 보관 정책과 이유 (재학습에 필요한 해상도):", "- 재전송이 실제로 있었는지, 어느 센서에서:", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=None, help="기본: .env 의 DB_PATH")
    ap.add_argument("--policy", choices=list(POLICIES), default="raw90_5m")
    ap.add_argument("--apply", action="store_true", help="집계 표를 채운다 (원본은 지우지 않음)")
    ap.add_argument("--delete", action="store_true", help="--apply 와 함께: 확인을 통과한 오래된 원본을 지운다")
    args = ap.parse_args()
    path = args.db or db_path()
    if not path.exists():
        print(f"{path} 가 없습니다. 수집기(python -m tsinfer.collector)가 쓰는 DB_PATH 를 확인하세요.")
        return 1
    policy = POLICIES[args.policy]
    if args.delete and not args.apply:
        ap.error("--delete 는 --apply 와 함께만 쓴다")
    if args.apply:
        print(apply(path, policy, delete=args.delete))
        return 0
    a = analyze(path)
    conn = connect_ro(path)
    try:
        pl = plan(conn, policy, a["now"])
    finally:
        conn.close()
    text = report(a, project(a), pl)
    out = ROOT / "reports" / "01b_storage.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
