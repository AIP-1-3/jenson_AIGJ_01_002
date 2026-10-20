"""매분 한 번 도는 추론 한 주기 (과제 8-1).

입력: 수집기 SQLite 의 최근 데이터 → 학습과 같은 preprocess.to_wide (과제 8-3)
출력: 공간마다 예측·이상·상태를 담은 dict (형식은 docs/interface.md)

모델은 predictors.py 의 등록부에서 고른다. 학습 모델 파일이 없으면 기준선(추세 외삽)으로
돌기 때문에, 모델을 만들기 전에도 서비스는 상시로 돈다.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.data.preprocess import clean, to_wide
from tsinfer.service import rules
from tsinfer.service.predictors import Forecaster

SCHEMA_VERSION = "1.0"


@dataclass
class EngineConfig:
    db_path: Path
    lookback: timedelta = timedelta(hours=6)
    stale_after: timedelta = timedelta(minutes=10)   # 마지막 수신 후 이 시간이 지나면 '데이터 끊김'


def read_recent(db_path: Path, since: datetime) -> pd.DataFrame:
    """수집기 DB 에서 최근 데이터를 긴 형식으로. 읽기 전용으로 열어 수집기와 부딪히지 않는다."""
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=10) as conn:
        df = pd.read_sql_query(
            "SELECT ts, space, device, metric, value, received_ts FROM readings WHERE ts >= ?",
            conn, params=(since.isoformat(),),
        )
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["received_ts"] = pd.to_datetime(df["received_ts"], utc=True, format="ISO8601")
    return df


def _round(x: Any, nd: int = 1) -> Any:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


class Engine:
    def __init__(self, cfg: dict[str, Any], ecfg: EngineConfig, forecaster: Forecaster):
        self.cfg = cfg
        self.ecfg = ecfg
        self.forecaster = forecaster

    def run_once(self, now: datetime | None = None, raw: pd.DataFrame | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        if raw is None:
            raw = read_recent(self.ecfg.db_path, now - self.ecfg.lookback)
        cleaned, rep = clean(raw, self.cfg)
        wide = to_wide(cleaned, self.cfg) if len(cleaned) else pd.DataFrame()
        # 막 시작된 5분 칸은 1~2분치만 있어 값이 흔들린다 → 완성된 칸까지만 쓴다 (과제 8 '생각해 볼 점')
        step = pd.Timedelta(self.cfg["preprocess"]["resample"])
        last_complete = pd.Timestamp(now).floor(step) - step
        if len(wide):
            wide = wide[wide.index <= last_complete]

        th = self.cfg["thresholds"]
        out: dict[str, Any] = {}
        for space in self.cfg["spaces"]:
            out[space] = self._space(space, raw, wide, now, th)
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now.isoformat(),
            "model": self.forecaster.name,
            "spaces": out,
            "removed_out_of_range": rep.removed_out_of_range,
        }

    def _space(self, space: str, raw: pd.DataFrame, wide: pd.DataFrame, now: datetime, th: dict) -> dict[str, Any]:
        sraw = raw[raw["space"] == space]
        last_rx = sraw["received_ts"].max() if len(sraw) else None
        stale = last_rx is None or (now - last_rx.to_pydatetime()) > self.ecfg.stale_after

        series = {m: wide[(space, m)].dropna() if (space, m) in wide.columns else pd.Series(dtype=float)
                  for m in self.cfg["metrics"]}
        current = {m: _round(s.iloc[-1]) if len(s) else None for m, s in series.items()}

        fc = self.forecaster.predict(wide, space) if not stale else {}
        horizons = self.cfg["forecast"]["report_horizons"]
        step_min = self.cfg["forecast"]["step_minutes"]
        all_h = list(range(step_min, self.cfg["forecast"]["horizon_minutes"] + 1, step_min))
        forecast = {
            m: {str(h): _round(fc[m].get(h)) for h in all_h} for m in ("co2", "temperature") if m in fc
        }
        co2_fc = fc.get("co2", {})
        minutes_to_vent = next((h for h in all_h if (co2_fc.get(h) or 0) >= th["co2_ventilate_ppm"]), None)

        alerts = rules.check(sraw, series, now, th, stale)
        occupied = rules.occupied(series.get("co2", pd.Series(dtype=float)))
        vent_alert = (
            (current.get("co2") or 0) < th["co2_ventilate_ppm"]
            and minutes_to_vent is not None
            and minutes_to_vent <= th["ventilation_alert_within_minutes"]
        )
        return {
            "data_ok": not stale,
            "last_received": last_rx.isoformat() if last_rx is not None else None,
            "current": current,
            "forecast": forecast,                       # {"co2": {"5": 812.3, ..., "60": 990.1}, ...}
            "co2_ventilate_in_minutes": minutes_to_vent,  # 1,000ppm 에 처음 닿는 예측 시점(분), 없으면 null
            "anomaly": {
                "is_anomaly": any(a["severity"] == "high" for a in alerts),
                "alerts": alerts,                       # 규칙 경보: data_gap / stuck / out_of_range / spike
            },
            "state": {
                "occupied": occupied,
                "ventilation_alert": bool(vent_alert),
            },
            "horizons_reported": horizons,
        }
