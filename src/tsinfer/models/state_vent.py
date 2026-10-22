"""과제 5 개선(C): 환기 경보에 '어제·지난주 같은 시각' 입력을 더해 재현율 85% 와 규칙 초과 macro-F1 을 함께 노린다.

    python -m tsinfer.models.state_vent      # → reports/05c_state_vent.md (models/forecast_gru.pt 재사용)

05b 4절과 같은 비교(규칙 / 기존 hgb / 예측 기반 / 결합 hgb)를 입력만 바꿔 다시 한다.
더하는 입력 (모두 예측 시점에 이미 지난 값만 쓴다)
  lag1d_now, lag1d_max30, lag1d_rise30 : 24시간 전 CO₂, 24시간 전부터 30분 동안의 최댓값, 그 사이 상승폭
  lag7d_now, lag7d_max30, lag7d_rise30 : 같은 것을 7일 전으로 (월요일처럼 전날이 휴일인 날을 덮는다)
  today_max                            : 오늘(한국 날짜) 0시부터 지금까지 최댓값
  min_to_th                            : 1000ppm 까지 남은 양 ÷ 최근 15분 상승 속도 (분, 안 오르면 큰 값)
운영점: 학습+검증 시간순 교차 예측에서 '재현율 ≥ 목표 중 macro-F1 최대'. 목표는 과제 기준 85% 에 2%p 여유를 둔 87%
(교차 예측과 시험 기간의 재현율이 몇 %p 흔들리기 때문). 시험 기간은 채점에만 쓴다.
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import state as st
from tsinfer.models import state_deep as sd

REPORT = ROOT / "reports" / "05c_state_vent.md"
RECALL_MARGIN_TARGET = 0.87
NEW_FEATURES = ("lag1d_now", "lag1d_max30", "lag1d_rise30", "lag7d_now", "lag7d_max30", "lag7d_rise30",
                "today_max", "min_to_th")


def seasonal_features(wide: pd.DataFrame, space: str, tz: str, within_min: int, threshold: float,
                      step_min: int = 5) -> pd.DataFrame:
    s = wide[(space, "co2")]
    idx = wide.index
    f = pd.DataFrame(index=idx)
    for name, lag in (("lag1d", pd.Timedelta(days=1)), ("lag7d", pd.Timedelta(days=7))):
        now = s.reindex(idx - lag).to_numpy(dtype=float)
        ahead = np.stack([s.reindex(idx - lag + pd.Timedelta(minutes=k)).to_numpy(dtype=float)
                          for k in range(step_min, within_min + 1, step_min)], axis=1)
        mx = pd.DataFrame(ahead).max(axis=1, skipna=True).to_numpy()       # 모두 비면 NaN
        f[f"{name}_now"] = now
        f[f"{name}_max30"] = mx
        f[f"{name}_rise30"] = mx - now
    day = idx.tz_convert(tz).normalize()
    f["today_max"] = s.groupby(day).cummax().to_numpy()
    rate = (s - s.shift(15 // step_min)) / 15.0                                   # ppm/분
    f["min_to_th"] = np.where(rate > 0.1, (threshold - s) / rate.clip(lower=0.1), 600.0)
    return f


def add_features(df: pd.DataFrame, wide: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    tz = cfg.get("timezone", "Asia/Seoul")
    th = cfg["thresholds"]
    step = cfg["forecast"]["step_minutes"]
    parts = []
    for sp in df["space"].unique():
        f = seasonal_features(wide, sp, tz, th["ventilation_alert_within_minutes"], th["co2_ventilate_ppm"], step)
        parts.append(f.rename_axis("time").reset_index().assign(space=sp))
    out = df.merge(pd.concat(parts, ignore_index=True), on=["time", "space"], how="left")
    out.index = df.index                       # 05b 의 구간 마스크(_parts)와 행을 맞춘다
    return out


def pass_lines(t: pd.DataFrame, title: str) -> list[str]:
    if t.empty or "pass" not in t:
        return [f"- {title}: 결과 없음"]
    ok = [m for m in t.index if pd.notna(t.loc[m, "pass"]) and bool(t.loc[m, "pass"])]
    return [f"- {title}: 기준(재현율 ≥85% 이고 규칙 macro-F1 초과) 만족 — " + (", ".join(
        f"{m} (F1 {t.loc[m, 'macro_f1']:.3f}, 재현율 {t.loc[m, 'recall']:.3f})" for m in ok) if ok else "없음")]


def build_report(cfg: dict[str, Any], before: dict, after: dict, target: float) -> list[str]:
    th = cfg["thresholds"]
    cols = [c for c in sd.VENT_COLS + ["note"]]
    lines = ["# 과제 5 개선(C). 환기 경보 — 어제·지난주 같은 시각 입력 추가", "",
             "## 측정 조건",
             f"- 양성: 지금 <{th['co2_ventilate_ppm']}ppm 이고 {th['ventilation_alert_within_minutes']}분 안에 "
             f"≥{th['co2_ventilate_ppm']}ppm (05b 와 같음). 시험 시점 {after['info'].get('test_rows')}개 "
             f"(양성 {after['info'].get('test_pos')}), 두 표가 같은 시점",
             f"- 더한 입력: {', '.join(NEW_FEATURES)} — 모두 예측 시점 이전 값",
             f"- 운영점: 학습+검증 시간순 교차 예측에서 재현율 ≥{target:.0%} 중 macro-F1 최대 "
             "(과제 기준 85% 에 여유를 둠). F1 최대 운영점도 함께 보고",
             f"- 예측: {after['info'].get('gru')}",
             "- 공정성: 입력·운영점 규칙은 실행 전에 정했고 임계값은 학습+검증 기간에서만 정함. 시험 기간은 채점에만 씀",
             "", "## 1. 기준 확인", "",
             *pass_lines(before["table"], "05b 입력 (같은 운영점 규칙)"),
             *pass_lines(after["table"], "05c 입력 (어제·지난주 추가)"), "",
             "## 2. 05c 입력 — 방법별", "",
             after["table"].reindex(columns=[c for c in cols if c in after["table"].columns]).to_markdown()
             if not after["table"].empty else "(없음)", "",
             "## 3. 05b 입력 — 방법별 (같은 운영점 규칙으로 다시 계산)", "",
             before["table"].reindex(columns=[c for c in cols if c in before["table"].columns]).to_markdown()
             if not before["table"].empty else "(없음)", "",
             "## 해석 (직접 작성)",
             "- 어제·지난주 같은 시각 정보가 경보에 도움이 된 이유(또는 안 된 이유):",
             "- 재현율 85% 를 지키면서 헛경보를 줄이려면 무엇이 더 필요한가 (예약표·PIR 등):", ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recall-target", type=float, default=RECALL_MARGIN_TARGET)
    args = ap.parse_args()

    from tsinfer.data.preprocess import clean, load_raw, to_wide

    cfg = load_yaml("default.yaml")
    sched = load_yaml("schedule.yaml")
    obs_path = ROOT / (sched.get("observations_csv") or "data/observations.csv")
    observations = pd.read_csv(obs_path) if obs_path.exists() else None
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    df = st.dataset(wide, cfg, sched, observations)
    before = sd.forecast_alarm(cfg, wide, df, recall_target=args.recall_target)
    after = sd.forecast_alarm(cfg, wide, add_features(df, wide, cfg), recall_target=args.recall_target)
    lines = build_report(cfg, before, after, args.recall_target)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
