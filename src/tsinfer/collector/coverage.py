"""수집 가동 보고 (과제 1 동작 근거): python -m tsinfer.collector.coverage [--hours 24]

SQLite readings 를 수신 시각(received_ts) 기준으로 묶어, 수집기가 실제로 받고 있었는지 본다.
- 공간별: 구간(기본 10분, 회의실 센서가 10분 주기라서) 중 1건 이상 받은 비율, 가장 긴 공백
- 전체: 아무 메시지도 받지 못한 공백 목록 (중단 시험 시각과 맞춰 보는 용도)
- 장치별: 저장 건수
수집기 중단 시험(scripts/outage_test.sh)의 공백이 여기 '공백 목록'에 나타나야 한다.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from tsinfer.config import db_path

TARGET = 0.95


def load(path: Path, since: datetime) -> pd.DataFrame:
    # 수집기가 쓰는 중에도 읽기만 하도록 읽기 전용으로 연다
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        df = pd.read_sql_query(
            "SELECT space, device, metric, received_ts FROM readings WHERE received_ts >= ?",
            conn,
            params=(since.isoformat(),),
        )
    finally:
        conn.close()
    df["received_ts"] = pd.to_datetime(df["received_ts"], utc=True, format="ISO8601")
    return df


def gaps(ts: pd.Series, start: pd.Timestamp, end: pd.Timestamp, min_gap: float) -> list[dict]:
    """시작·끝 경계를 포함해 min_gap 분 이상 비어 있던 구간."""
    t = pd.concat([pd.Series([start]), ts.sort_values(), pd.Series([end])], ignore_index=True)
    d = t.diff().dt.total_seconds().div(60)
    out = []
    for i in d.index[d >= min_gap]:
        out.append({"from": t[i - 1].isoformat(), "to": t[i].isoformat(), "minutes": round(float(d[i]), 1)})
    return out


def coverage(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, bin_min: int, min_gap: float) -> dict:
    bins = pd.date_range(start.floor(f"{bin_min}min"), end, freq=f"{bin_min}min", inclusive="left")
    n_bins = max(len(bins), 1)
    spaces = {}
    for space, g in df.groupby("space"):
        hit = g["received_ts"].dt.floor(f"{bin_min}min").nunique()
        sp_gaps = gaps(g["received_ts"], start, end, min_gap)
        spaces[space] = {
            "rows": int(len(g)),
            "coverage": round(min(hit / n_bins, 1.0), 4),
            "longest_gap_min": max((x["minutes"] for x in sp_gaps), default=0.0),
            "pass": hit / n_bins >= TARGET,
        }
    devices = df.groupby(["space", "device"]).size().rename("rows").reset_index()
    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "bin_min": bin_min,
        "target": TARGET,
        "rows": int(len(df)),
        "spaces": spaces,
        "devices": devices.to_dict("records"),
        "gaps_all": gaps(df["received_ts"], start, end, min_gap) if len(df) else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--bin", type=int, default=10, help="구간 길이(분)")
    ap.add_argument("--min-gap", type=float, default=5, help="공백으로 볼 최소 분")
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args()
    end = pd.Timestamp(datetime.now(timezone.utc))
    start = end - pd.Timedelta(hours=args.hours)
    df = load(args.db or db_path(), start.to_pydatetime())
    print(json.dumps(coverage(df, start, end, args.bin, args.min_gap), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
