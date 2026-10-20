"""과제 2-5: 기준선 3종의 5·15·30·60분 뒤 MAE.

    python -m tsinfer.data.baselines            # 시험 기간 기준, reports/02_baselines.md 저장
    python -m tsinfer.data.baselines --period valid

기준선 (시점 t 에서 t+h 를 예측):
  persistence : 지금 값이 그대로        ŷ(t+h) = y(t)
  mean_1h     : 최근 1시간 평균          ŷ(t+h) = mean(y(t-55분 … t))
  yesterday   : 어제 같은 시각            ŷ(t+h) = y(t+h-24시간)

이후 모델(과제 3)은 evaluate() 에 같은 eval_index 를 넘겨 '같은 시점, 같은 조건'으로 비교한다.
"""
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.data.preprocess import clean, load_raw, to_wide
from tsinfer.data.split import split_by_time

Predictor = Callable[[pd.Series, int, int], pd.Series]


def persistence(s: pd.Series, h: int, steps_per_day: int) -> pd.Series:
    return s


def mean_1h(s: pd.Series, h: int, steps_per_day: int) -> pd.Series:
    w = max(steps_per_day // 24, 1)
    return s.rolling(w, min_periods=max(w // 2, 1)).mean()


def yesterday(s: pd.Series, h: int, steps_per_day: int) -> pd.Series:
    # t 시점의 예측값 = (t+h) 의 24시간 전 값 = s 를 (하루 - h) 만큼 뒤로 민 것
    return s.shift(steps_per_day - h)


BASELINES: dict[str, Predictor] = {"persistence": persistence, "mean_1h": mean_1h, "yesterday": yesterday}


def evaluate(wide: pd.DataFrame, eval_index: pd.DatetimeIndex, horizons_min: list[int],
             predictors: dict[str, Predictor] = BASELINES) -> pd.DataFrame:
    """(space, metric, method, horizon) 별 MAE.

    전체 시계열(wide)로 예측을 만든 뒤 eval_index 시점에서만 채점한다.
    모든 방법이 값을 낸 시점만 써서 방법 간 비교 조건을 맞춘다.
    """
    step = pd.Timedelta(pd.infer_freq(wide.index[:50]) or "5min")
    steps_per_day = int(pd.Timedelta("1D") / step)
    rows = []
    for col in wide.columns:
        s = wide[col]
        for hm in horizons_min:
            h = int(pd.Timedelta(minutes=hm) / step)
            target = s.shift(-h)
            preds = {name: f(s, h, steps_per_day) for name, f in predictors.items()}
            frame = pd.DataFrame({"y": target, **preds}).loc[lambda x: x.index.isin(eval_index)].dropna()
            for name in predictors:
                rows.append({
                    "space": col[0], "metric": col[1], "method": name, "horizon_min": hm,
                    "mae": float(np.abs(frame[name] - frame["y"]).mean()) if len(frame) else np.nan,
                    "n": len(frame),
                })
    return pd.DataFrame(rows)


def to_table(res: pd.DataFrame) -> pd.DataFrame:
    return res.pivot_table(index=["space", "metric", "method"], columns="horizon_min", values="mae").round(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", choices=["valid", "test"], default="test")
    args = ap.parse_args()

    cfg = load_yaml("default.yaml")
    raw = load_raw(metrics=cfg["metrics"])
    cleaned, rep = clean(raw, cfg)
    wide = to_wide(cleaned, cfg)
    train, valid, test = split_by_time(wide, cfg)
    period = valid if args.period == "valid" else test
    res = evaluate(wide, period.index, cfg["forecast"]["report_horizons"])
    table = to_table(res)

    out = ROOT / "reports" / "02_baselines.md"
    lines = [
        "# 과제 2. 기준선 성능표 (MAE)",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터 InfluxDB iot-bucket, {wide.index.min():%Y-%m-%d} ~ {wide.index.max():%Y-%m-%d} (UTC)",
        f"- 평가 기간({args.period}): {period.index.min():%Y-%m-%d} ~ {period.index.max():%Y-%m-%d}",
        f"- 분석 간격: {cfg['preprocess']['resample']}, 같은 공간 장치들은 평균",
        f"- 제거한 범위 밖 값: {rep.removed_out_of_range} (전체 {rep.rows_in}행)",
        "",
        "## 결과 (단위: CO₂ ppm, 온도 ℃, 습도 %)",
        "",
        table.to_markdown(),
        "",
    ]
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string())
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
