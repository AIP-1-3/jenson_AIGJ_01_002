"""12시간 무중단 보고 (과제 8-4): python -m tsinfer.service.summary [--hours 12]

logs/service_cycles.jsonl 에서 주기 수, 실패, 최대 공백(주기 사이 간격), 공간별 끊김 경보 구간을 센다.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from tsinfer.config import ROOT


def summarize(path: Path, hours: float, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return {"cycles": 0}
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df = df[df["ts"] >= pd.Timestamp(now - timedelta(hours=hours))].sort_values("ts")
    if df.empty:
        return {"cycles": 0}
    gaps = df["ts"].diff().dt.total_seconds().div(60).dropna()
    expected = int((df["ts"].iloc[-1] - df["ts"].iloc[0]).total_seconds() // 60) + 1
    out = {
        "from": df["ts"].iloc[0].isoformat(),
        "to": df["ts"].iloc[-1].isoformat(),
        "cycles": int(len(df)),
        "expected_cycles": expected,
        "failed_cycles": int((~df["ok"].astype(bool)).sum()),
        "max_gap_min": round(float(gaps.max()), 1) if len(gaps) else 0.0,
        "p95_duration_ms": round(float(df["duration_ms"].quantile(0.95)), 1),
    }
    if "published" in df:
        out["publish_failed"] = int((df["published"] == False).sum())  # noqa: E712
    ok = df[df["ok"].astype(bool)]
    if "data_ok" in ok and len(ok):
        gap_minutes = {}
        for d in ok["data_ok"]:
            for space, good in d.items():
                gap_minutes[space] = gap_minutes.get(space, 0) + (0 if good else 1)
        out["data_gap_minutes"] = gap_minutes
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12)
    ap.add_argument("--log", type=Path, default=ROOT / "logs" / "service_cycles.jsonl")
    args = ap.parse_args()
    print(json.dumps(summarize(args.log, args.hours), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
