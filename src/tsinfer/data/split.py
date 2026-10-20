"""과제 2-3: 시간순 분할. 무작위로 섞으면 바로 옆 시점이 학습·시험에 동시에 들어가
미래를 엿본 점수가 나온다. 정규화 통계는 학습 기간으로만 계산한다."""
from __future__ import annotations

from typing import Any

import pandas as pd


def split_by_time(wide: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sp = cfg["split"]
    if not sp.get("train_end") or not sp.get("valid_end"):
        raise ValueError("configs/default.yaml 의 split.train_end, split.valid_end 를 정하세요.")
    t_end = pd.Timestamp(sp["train_end"], tz="UTC")
    v_end = pd.Timestamp(sp["valid_end"], tz="UTC")
    if not t_end < v_end:
        raise ValueError("train_end 는 valid_end 보다 앞이어야 합니다.")
    return wide[wide.index < t_end], wide[(wide.index >= t_end) & (wide.index < v_end)], wide[wide.index >= v_end]


def fit_scaler(train: pd.DataFrame) -> pd.DataFrame:
    """열별 평균·표준편차 (학습 기간만). 표준편차 0 은 1 로 둬서 나눗셈 오류를 막는다."""
    stats = pd.DataFrame({"mean": train.mean(), "std": train.std()})
    stats["std"] = stats["std"].replace(0, 1).fillna(1)
    return stats


def apply_scaler(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    return (df - stats["mean"]) / stats["std"]
