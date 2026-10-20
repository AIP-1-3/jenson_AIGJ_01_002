"""규칙 기반 판단 — 과제 4(규칙 탐지)·과제 5(단순 규칙 기준선)·과제 8(규칙 경보)에서 같이 쓴다."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

# 같은 값이 이 시간(분) 이상 이어지면 '값 고정' (고장 정의: 1~4시간).
# 온도·습도는 밤에 0.1 단위로 한 시간쯤 그대로일 수 있어 길게 잡는다 (헛경보 줄이기).
STUCK_MIN = {"co2": 60, "temperature": 180, "humidity": 180}
SPIKE = {"co2": 300.0, "temperature": 3.0, "humidity": 15.0}   # 5분 간격 한 칸 변화량
OCCUPIED_RISE_PPM = 30.0  # 과제 5 단순 규칙: 최근 15분 CO₂ 상승 ≥ 30ppm → 사용중


def _alert(kind: str, metric: str | None, severity: str, detail: str) -> dict[str, Any]:
    return {"type": kind, "metric": metric, "severity": severity, "detail": detail}


def check(sraw: pd.DataFrame, series: dict[str, pd.Series], now: datetime, th: dict, stale: bool) -> list[dict]:
    alerts: list[dict] = []
    if stale:
        last = sraw["received_ts"].max() if len(sraw) else None
        alerts.append(_alert("data_gap", None, "high",
                             f"마지막 수신 {last.isoformat() if last is not None else '없음'}"))

    # 값 고정: 장치 원본 기준 (평균을 내면 한 장치가 멈춰도 가려진다)
    for (dev, metric), g in sraw.groupby(["device", "metric"]):
        if metric not in STUCK_MIN:
            continue
        g = g[g["ts"] >= pd.Timestamp(now - timedelta(minutes=STUCK_MIN[metric]))]
        if len(g) < 6:
            continue
        span = (g["ts"].max() - g["ts"].min()).total_seconds() / 60
        if g["value"].nunique() == 1 and span >= STUCK_MIN[metric] * 0.8:
            alerts.append(_alert("stuck", metric, "high", f"{dev}: {span:.0f}분 동안 {g['value'].iloc[0]} 고정"))

    # 범위 밖 (clean 에서 지워지기 전 원본 기준, 최근 1시간)
    recent = sraw[sraw["ts"] >= pd.Timestamp(now - timedelta(minutes=60))]
    for metric, lo, hi in (("co2", 350, 5000), ("temperature", -10, 50), ("humidity", 0, 100)):
        bad = recent[(recent["metric"] == metric) & ~recent["value"].between(lo, hi)]
        if len(bad):
            alerts.append(_alert("out_of_range", metric, "high",
                                 f"{bad['device'].iloc[-1]}: {bad['value'].iloc[-1]} ({len(bad)}건)"))

    # 급변: 공간 시계열 마지막 한 칸
    for metric, s in series.items():
        if metric in SPIKE and len(s) >= 2:
            d = float(s.iloc[-1] - s.iloc[-2])
            if abs(d) >= SPIKE[metric]:
                alerts.append(_alert("spike", metric, "medium", f"5분 사이 {d:+.1f}"))

    co2 = series.get("co2")
    if co2 is not None and len(co2) and co2.iloc[-1] >= th["co2_critical_ppm"]:
        alerts.append(_alert("co2_critical", "co2", "high", f"현재 {co2.iloc[-1]:.0f}ppm"))
    return alerts


def occupied(co2: pd.Series, window_steps: int = 3) -> bool | None:
    """과제 5 단순 규칙. 데이터가 모자라면 None (모름)."""
    if len(co2) <= window_steps or not np.isfinite(co2.iloc[-1]):
        return None
    return bool(co2.iloc[-1] - co2.iloc[-1 - window_steps] >= OCCUPIED_RISE_PPM)
