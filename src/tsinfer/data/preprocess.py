"""과제 2-1·2-2: 원본(긴 형식) → 정제 → 공간·항목별 등간격 시계열(넓은 형식).

학습·평가·서비스(과제 8)가 모두 이 함수를 써야 전처리가 어긋나지 않는다.

입력 긴 형식 컬럼: ts, space, device, metric, value (influx_export·수집기 공통)
출력 넓은 형식: index=ts(UTC, 등간격), columns=MultiIndex(space, metric)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from tsinfer.config import ROOT


@dataclass
class CleanReport:
    """진단서에 그대로 옮길 숫자 (과제 2-4)."""

    rows_in: int = 0
    removed_out_of_range: dict[str, int] = field(default_factory=dict)
    rows_out: int = 0


def load_raw(raw_dir: Path | None = None, metrics: Iterable[str] | None = None) -> pd.DataFrame:
    """data/raw/influx_*.parquet 를 읽어 하나의 긴 형식 DataFrame 으로."""
    raw_dir = raw_dir or ROOT / "data" / "raw"
    files = sorted(raw_dir.glob("influx_*.parquet"))
    if metrics is not None:
        wanted = set(metrics)
        files = [f for f in files if f.name.split("_")[1] in wanted]
    if not files:
        raise FileNotFoundError(f"{raw_dir} 에 parquet 가 없습니다. 먼저 tsinfer.ingest.influx_export 를 실행하세요.")
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop_duplicates(["ts", "space", "device", "metric"])


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, CleanReport]:
    """물리적으로 불가능한 값 제거 (0ppm, 60℃ 등). 범위는 configs/default.yaml."""
    rep = CleanReport(rows_in=len(df))
    ranges: dict[str, list[float]] = cfg["preprocess"]["valid_range"]
    keep = pd.Series(True, index=df.index)
    for metric, (lo, hi) in ranges.items():
        bad = (df["metric"] == metric) & ~df["value"].between(lo, hi)
        rep.removed_out_of_range[metric] = int(bad.sum())
        keep &= ~bad
    out = df[keep]
    rep.rows_out = len(out)
    return out, rep


def to_wide(df: pd.DataFrame, cfg: dict[str, Any], spaces: list[str] | None = None,
            metrics: list[str] | None = None) -> pd.DataFrame:
    """공간·항목별 등간격 시계열.

    1) 장치별로 1분 평균 → 2) 같은 공간의 장치들을 평균 → 3) 분석 간격(기본 5분)으로 평균
    4) short_gap_max_minutes 이하 결측만 선형 보간, 그보다 긴 결측은 NaN 으로 남긴다.
    (긴 결측을 메우면 '값 고정' 고장과 구별할 수 없게 된다.)
    """
    pp = cfg["preprocess"]
    spaces = spaces or cfg["spaces"]
    metrics = metrics or cfg["metrics"]
    freq = pp["resample"]
    step_min = pd.Timedelta(freq).total_seconds() / 60
    gap_limit = max(int(pp["short_gap_max_minutes"] // step_min), 0)

    d = df[df["space"].isin(spaces) & df["metric"].isin(metrics)]
    per_dev = (
        d.set_index("ts")
        .groupby(["space", "metric", "device"])["value"]
        .resample("1min").mean()
        .reset_index()
    )
    per_space = per_dev.groupby(["ts", "space", "metric"])["value"].mean().unstack(["space", "metric"])
    wide = per_space.resample(freq).mean()
    if gap_limit:
        wide = wide.interpolate(method="time", limit=gap_limit, limit_area="inside")
    cols = pd.MultiIndex.from_product([spaces, metrics], names=["space", "metric"])
    return wide.reindex(columns=cols).sort_index()


def coverage(wide: pd.DataFrame) -> pd.DataFrame:
    """공간·항목별 결측률(%) — 진단서 표."""
    return (wide.isna().mean() * 100).round(1).unstack("metric")
