"""예측기 등록부. 서비스는 이 인터페이스만 알면 된다.

predict(wide, space) → {"co2": {5: 812.3, 10: ..., 60: ...}, "temperature": {...}}
- wide: preprocess.to_wide 결과 (학습과 같은 전처리)
- 과제 3 모델이 생기면 여기에 클래스를 추가하고 load_forecaster 에서 고른다.
"""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

TARGETS = ("co2", "temperature")


class Forecaster(Protocol):
    name: str

    def predict(self, wide: pd.DataFrame, space: str) -> dict[str, dict[int, float]]: ...


class TrendBaseline:
    """기준선 v0: 최근 30분 기울기를 외삽하되, 너무 멀리 뻗지 않게 감쇠한다.
    '현재값 유지'와 달리 1,000ppm 도달 시점을 미리 낼 수 있어 서비스 경보에 쓸 만하다."""

    name = "trend_baseline_v0"

    def __init__(self, cfg: dict[str, Any], window_min: int = 30, damping: float = 0.97):
        self.step = cfg["forecast"]["step_minutes"]
        self.horizons = list(range(self.step, cfg["forecast"]["horizon_minutes"] + 1, self.step))
        self.window = max(window_min // self.step, 2)
        self.damping = damping

    def predict(self, wide: pd.DataFrame, space: str) -> dict[str, dict[int, float]]:
        out: dict[str, dict[int, float]] = {}
        for m in TARGETS:
            if (space, m) not in wide.columns:
                continue
            s = wide[(space, m)].dropna().iloc[-(self.window + 1):]
            if len(s) < 2:
                continue
            slope = (s.iloc[-1] - s.iloc[0]) / (len(s) - 1)       # 한 칸(5분)당 변화량
            last, acc, preds = float(s.iloc[-1]), 0.0, {}
            for i, h in enumerate(self.horizons, start=1):
                acc += slope * self.damping ** i
                preds[h] = last + acc
            if m == "co2":   # 외기(약 420ppm) 아래로는 내려가지 않는다
                preds = {h: max(v, 400.0) for h, v in preds.items()}
            out[m] = preds
        return out


class Persistence:
    name = "persistence"

    def __init__(self, cfg: dict[str, Any]):
        step = cfg["forecast"]["step_minutes"]
        self.horizons = list(range(step, cfg["forecast"]["horizon_minutes"] + 1, step))

    def predict(self, wide: pd.DataFrame, space: str) -> dict[str, dict[int, float]]:
        out = {}
        for m in TARGETS:
            if (space, m) in wide.columns:
                s = wide[(space, m)].dropna()
                if len(s) and np.isfinite(s.iloc[-1]):
                    out[m] = {h: float(s.iloc[-1]) for h in self.horizons}
        return out


def load_forecaster(cfg: dict[str, Any], name: str | None = None) -> Forecaster:
    name = name or "trend"
    if name == "persistence":
        return Persistence(cfg)
    return TrendBaseline(cfg)
