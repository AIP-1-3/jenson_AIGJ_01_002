"""예측기 등록부. 서비스는 이 인터페이스만 알면 된다.

predict(wide, space) → {"co2": {5: 812.3, 10: ..., 60: ...}, "temperature": {...}}
- wide: preprocess.to_wide 결과 (학습과 같은 전처리)
- 과제 3 모델이 생기면 여기에 클래스를 추가하고 load_forecaster 에서 고른다.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

TARGETS = ("co2", "temperature")
log = logging.getLogger("predictors")


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


TAIL_FILL_STEPS = 2   # 10분 = preprocess.short_gap_max_minutes 와 같은 한도


def fill_tail(wide: pd.DataFrame, steps: int = TAIL_FILL_STEPS) -> pd.DataFrame:
    """맨 끝 칸들의 결측만 직전 값으로 채운다 (최대 steps 칸).
    학습 때는 짧은 결측을 앞뒤 값으로 보간하지만, 서비스에서는 '뒤' 값이 아직 없어 끝 칸이 비고
    창 전체가 버려진다. 10분 주기 센서 한 대뿐인 회의실은 두 칸에 한 번 이렇게 예측이 비었다.
    미래를 보지 않도록 앞으로만 채우고, 한도를 넘는 결측(진짜 끊김)은 그대로 둔다."""
    if steps <= 0 or len(wide) == 0:
        return wide
    out = wide.copy()
    tail = out.iloc[-(steps + 1):]
    out.iloc[-(steps + 1):] = tail.ffill(limit=steps).to_numpy()
    return out


class GRUForecaster:
    """과제 3 모델. 학습과 같은 forecast.make_windows 로 입력을 만든다 (과제 8-3)."""

    def __init__(self, path=None):
        from tsinfer.models import forecast

        self._fc = forecast
        self.model, self.spec = forecast.load(path or forecast.MODEL_PATH)
        self.name = f"gru_v1(lookback={self.spec.lookback * self.spec.step_minutes}m)"

    def predict(self, wide: pd.DataFrame, space: str) -> dict[str, dict[int, float]]:
        if space not in self.spec.spaces or len(wide) < self.spec.lookback:
            return {}
        wide = fill_tail(wide)
        x, _, info = self._fc.make_windows(wide, self.spec, at=wide.index[-1:], with_target=False)
        info_sp = info["space"].to_numpy() == space
        if not info_sp.any():
            return {}   # 최근 창에 결측이 있으면 예측하지 않는다 (서비스는 기준선 없이 빈 예측)
        x, info = x[info_sp], info[info_sp].reset_index(drop=True)
        pred = self._fc.decode(self._fc.predict_array(self.model, x), info, self.spec)
        mins = [h * self.spec.step_minutes for h in self.spec.horizons]
        out = {m: {hm: float(v) for hm, v in zip(mins, pred[m][0])} for m in pred}
        if "co2" in out:
            out["co2"] = {h: max(v, 400.0) for h, v in out["co2"].items()}
        return out


def load_forecaster(cfg: dict[str, Any], name: str | None = None) -> Forecaster:
    name = name or "trend"
    if name == "persistence":
        return Persistence(cfg)
    if name == "gru":
        try:
            return GRUForecaster()
        except Exception as e:  # 모델 파일·torch 가 없으면 기준선으로라도 서비스는 돈다
            log.warning("GRU 모델을 못 불러와 추세 기준선으로 돕니다: %r", e)
    return TrendBaseline(cfg)
