"""InfluxDB 과거 데이터 → data/raw/*.parquet

    python -m tsinfer.ingest.influx_export --start 2026-07-01
    python -m tsinfer.ingest.influx_export --start 2026-07-01 --metrics co2 temperature

- iot-bucket 은 항목(co2, temperature …)마다 measurement 가 따로 있고, 값은 field "value",
  태그는 device_id / location / point / spot 이다 (2026-10-20 확인).
- 공용 서버 부하와 Jetson 메모리를 생각해 항목 × 7일 단위로 나눠 받는다.
- 이미 받은 조각은 건너뛰므로 중간에 끊겨도 같은 명령으로 이어 받으면 된다.
  (마지막 조각은 아직 쌓이는 중이라 매번 다시 받는다)
- 저장 형식은 수집기 SQLite 와 같은 긴 형식: ts, space, device, metric, value, point
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

from tsinfer.config import ROOT

log = logging.getLogger("influx_export")

DEFAULT_METRICS = [
    "co2", "temperature", "humidity", "pressure", "tvoc",
    "illumination", "activity", "LAeq", "battery",
]
# 수집기(sensors.yaml)와 같은 항목 이름으로 맞춘다
RENAME = {"activity": "pir_activity", "LAeq": "sound_laeq"}
COLUMNS = ["ts", "space", "device", "metric", "value", "point"]

FLUX = """
from(bucket: "{bucket}")
  |> range(start: {start}, stop: {stop})
  |> filter(fn: (r) => r._measurement == "{metric}" and r._field == "value")
  |> keep(columns: ["_time", "_value", "device_id", "location", "point"])
"""


def chunks(start: datetime, stop: datetime, days: int):
    cur = start
    while cur < stop:
        nxt = min(cur + timedelta(days=days), stop)
        yield cur, nxt
        cur = nxt


def fetch(client: InfluxDBClient, org: str, bucket: str, metric: str, start: datetime, stop: datetime) -> pd.DataFrame:
    flux = FLUX.format(bucket=bucket, metric=metric, start=start.isoformat(), stop=stop.isoformat())
    rows = [
        (r.get_time(), r.values.get("location"), r.values.get("device_id"), r.get_value(), r.values.get("point"))
        for r in client.query_api().query_stream(flux, org=org)
    ]
    df = pd.DataFrame(rows, columns=["ts", "space", "device", "value", "point"])
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df.insert(3, "metric", RENAME.get(metric, metric))
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])[COLUMNS]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--stop", help="YYYY-MM-DD (UTC), 기본: 지금")
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--chunk-days", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv(ROOT / ".env")
    url, token = os.environ["INFLUX_URL"], os.environ["INFLUX_TOKEN"]
    org, bucket = os.environ["INFLUX_ORG"], os.environ.get("INFLUX_BUCKET") or "iot-bucket"

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    stop = (
        datetime.fromisoformat(args.stop).replace(tzinfo=timezone.utc)
        if args.stop else datetime.now(timezone.utc)
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    with InfluxDBClient(url=url, token=token, org=org, timeout=120_000) as client:
        for metric in args.metrics:
            for a, b in chunks(start, stop, args.chunk_days):
                path = out / f"influx_{RENAME.get(metric, metric)}_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
                is_last = b >= stop
                if path.exists() and not is_last:
                    continue
                df = fetch(client, org, bucket, metric, a, b)
                if is_last:  # 아직 쌓이는 조각은 이전 파일을 대체
                    for old in out.glob(f"influx_{RENAME.get(metric, metric)}_{a:%Y%m%d}_*.parquet"):
                        old.unlink()
                df.to_parquet(path, index=False)
                total += len(df)
                log.info("%s %s~%s: %d행", metric, f"{a:%m-%d}", f"{b:%m-%d}", len(df))
    log.info("완료: %d행 → %s", total, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
