"""과제 2-4 데이터 진단서: python -m tsinfer.data.diagnose

reports/02_diagnosis.md 와 reports/figures/*.png 를 만든다. 숫자만 뽑고, 해석은 사람이 쓴다
(보고서 끝 '해석' 칸). 시간대·요일은 configs/default.yaml 의 timezone(한국 시간) 기준.

담는 것
1) 센서별 기록 주기 — 같은 공간에 주기가 다른 센서가 섞이면 평균이 계단처럼 흔들린다
2) 결측률(보간 전·후)과 긴 결측 구간 — 원인 추정의 출발점
3) 평일·주말 시간대별 CO₂ 평균 — 수업·회의 패턴
4) CO₂ 상승 사건의 반응 시간 — 사람이 들어온 뒤 정점까지, 정점 뒤 절반으로 떨어지기까지
5) 평소와 어긋난 날 — 평일 평균 하루 모양과 가장 다른 날
"""
from __future__ import annotations

import argparse
import copy
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.data.preprocess import clean, coverage, load_raw, to_wide

RISE_PPM = 30.0      # 15분(3칸) 동안 이만큼 오르면 상승 시작으로 본다 (서비스 사용 여부 규칙과 같은 값)
FLAT_PPM = 10.0      # 직전 15분 변화가 이보다 작아야 '시작'으로 인정 (이미 오르던 중이면 제외)
MIN_RISE_PPM = 80.0  # 시작~정점 상승폭이 이보다 작은 사건은 잡음으로 버린다


def device_intervals(cleaned: pd.DataFrame) -> pd.DataFrame:
    """공간·장치·항목별 기록 간격 중앙값(분)과 건수."""
    d = cleaned.sort_values("ts")
    gap = d.groupby(["space", "device", "metric"])["ts"].diff().dt.total_seconds() / 60
    out = (d.assign(gap=gap).groupby(["space", "device", "metric"])
           .agg(rows=("value", "size"), median_interval_min=("gap", "median"),
                first=("ts", "min"), last=("ts", "max")))
    out["median_interval_min"] = out["median_interval_min"].round(1)
    return out


def long_gaps(wide_raw: pd.DataFrame, min_minutes: int = 60, top: int = 15) -> pd.DataFrame:
    """보간 전 넓은 형식에서 min_minutes 이상 비어 있는 구간 (긴 순)."""
    step = wide_raw.index[1] - wide_raw.index[0] if len(wide_raw) > 1 else pd.Timedelta("5min")
    rows = []
    for col in wide_raw.columns:
        na = wide_raw[col].isna()
        if not na.any():
            continue
        grp = (na != na.shift()).cumsum()
        for _, run in wide_raw.index.to_series()[na].groupby(grp[na]):
            minutes = (run.iloc[-1] - run.iloc[0] + step).total_seconds() / 60
            if minutes >= min_minutes:
                rows.append({"space": col[0], "metric": col[1], "start": run.iloc[0],
                             "end": run.iloc[-1] + step, "hours": round(minutes / 60, 1)})
    df = pd.DataFrame(rows, columns=["space", "metric", "start", "end", "hours"])
    return df.sort_values("hours", ascending=False).head(top).reset_index(drop=True)


def hourly_profile(s: pd.Series, tz: str) -> pd.DataFrame:
    """평일·주말 한국 시간 시각별 평균."""
    local = s.dropna().tz_convert(tz)
    kind = np.where(local.index.dayofweek < 5, "평일", "주말")
    return local.groupby([kind, local.index.hour]).mean().unstack(0).round(0)


def rise_events(co2: pd.Series, tz: str, step_min: int = 5) -> pd.DataFrame:
    """CO₂ 상승 사건: 시작 시각, 시작값, 정점, 정점까지 분, 정점 뒤 절반 회복까지 분.

    정답 일정이 없으므로 '사람이 들어온 시점'을 CO₂ 가 오르기 시작한 칸으로 근사한다.
    실제 일정 대비 지연은 일정표를 받으면 같은 함수로 다시 잰다.
    """
    s = co2.dropna()
    if len(s) < 10:
        return pd.DataFrame()
    s = s.asfreq(f"{step_min}min")
    d3 = s - s.shift(3)
    prev = s.shift(3) - s.shift(6)
    starts = s.index[(d3 >= RISE_PPM) & (prev.abs() < FLAT_PPM)]
    rows, busy_until = [], s.index[0]
    for t in starts:
        if t <= busy_until:
            continue
        # 실제로 오르기 시작한 칸을 찾고, 그 직전 칸을 시작(기준값)으로 둔다
        look = s[t - pd.Timedelta(minutes=3 * step_min): t]
        rising = look.diff() > 3.0
        first = rising.idxmax() if rising.any() else t
        t0 = first - pd.Timedelta(minutes=step_min)
        base = s.get(t0)
        window = s[t0: t0 + pd.Timedelta(hours=4)]
        if base is None or np.isnan(base) or window.isna().mean() > 0.2:
            continue
        peak_t = window.idxmax()
        peak = window.max()
        if peak - base < MIN_RISE_PPM:
            continue
        after = s[peak_t: peak_t + pd.Timedelta(hours=6)]
        half = after[after <= base + (peak - base) / 2]
        rows.append({
            "start_local": first.tz_convert(tz),   # 처음 오른 칸
            "base_ppm": round(float(base)),
            "peak_ppm": round(float(peak)),
            "to_peak_min": (peak_t - first).total_seconds() / 60,
            "half_decay_min": (half.index[0] - peak_t).total_seconds() / 60 if len(half) else np.nan,
        })
        busy_until = max(peak_t, t + pd.Timedelta(minutes=3 * step_min))
    return pd.DataFrame(rows)


def odd_days(s: pd.Series, tz: str, top: int = 5) -> pd.DataFrame:
    """평일 평균 하루 모양(시각별)과 RMSE 가 가장 큰 평일. 데이터가 하루의 70% 미만인 날은 뺀다."""
    local = s.tz_convert(tz)
    local = local[local.index.dayofweek < 5]
    by_day = local.groupby([local.index.date, local.index.hour]).mean().unstack()
    full = local.groupby(local.index.date).apply(lambda x: x.notna().mean())
    by_day = by_day[full.reindex(by_day.index) >= 0.7]
    if by_day.empty:
        return pd.DataFrame(columns=["date", "rmse_ppm", "max_ppm", "mean_profile_max_ppm"])
    mean = by_day.mean()
    rmse = np.sqrt(((by_day - mean) ** 2).mean(axis=1))
    out = pd.DataFrame({"rmse_ppm": rmse.round(0), "max_ppm": by_day.max(axis=1).round(0)})
    out["mean_profile_max_ppm"] = round(float(mean.max()))
    out.index.name = "date"
    return out.sort_values("rmse_ppm", ascending=False).head(top).reset_index()


def _plot_profiles(profiles: dict[str, pd.DataFrame], path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    for font in ("NanumGothic", "Noto Sans CJK KR", "AppleGothic"):
        from matplotlib import font_manager
        if any(font in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = font
            break
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind in zip(axes, ("평일", "주말")):
        for space, prof in profiles.items():
            if kind in prof:
                ax.plot(prof.index, prof[kind], marker=".", label=space)
        ax.axhline(1000, color="gray", ls="--", lw=0.8)
        ax.set_title(f"CO2 by hour ({'weekday' if kind == '평일' else 'weekend'}, KST)")
        ax.set_xlabel("hour")
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("ppm")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(**kw) if len(df) else "(없음)"


def build(raw: pd.DataFrame, cfg: dict[str, Any]) -> tuple[list[str], dict[str, pd.DataFrame]]:
    tz = cfg.get("timezone", "Asia/Seoul")
    cleaned, rep = clean(raw, cfg)
    no_interp = copy.deepcopy(cfg)
    no_interp["preprocess"]["short_gap_max_minutes"] = 0
    wide_raw = to_wide(cleaned, no_interp)
    wide = to_wide(cleaned, cfg)

    miss = pd.concat({"보간 전": coverage(wide_raw), "보간 후": coverage(wide)}, axis=1)
    intervals = device_intervals(cleaned[cleaned["metric"].isin(cfg["metrics"])])
    gaps = long_gaps(wide_raw)
    co2_spaces = [sp for sp in cfg["spaces"] if (sp, "co2") in wide and wide[(sp, "co2")].notna().any()]
    profiles = {sp: hourly_profile(wide[(sp, "co2")], tz) for sp in co2_spaces}
    events = {sp: rise_events(wide[(sp, "co2")], tz) for sp in co2_spaces}
    odd = {sp: odd_days(wide[(sp, "co2")], tz) for sp in co2_spaces}

    ev_rows = []
    for sp, ev in events.items():
        if len(ev):
            ev_rows.append({"space": sp, "events": len(ev),
                            "rise_ppm_median": round(float((ev["peak_ppm"] - ev["base_ppm"]).median())),
                            "to_peak_min_median": float(ev["to_peak_min"].median()),
                            "half_decay_min_median": float(ev["half_decay_min"].median()),
                            "start_hour_mode": int(ev["start_local"].dt.hour.mode().iloc[0])})
    ev_summary = pd.DataFrame(ev_rows)

    prof_tbl = pd.concat({sp: p for sp, p in profiles.items()}, axis=1) if profiles else pd.DataFrame()
    lines = [
        "# 과제 2. 데이터 진단서",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터 InfluxDB iot-bucket, {wide.index.min():%Y-%m-%d} ~ {wide.index.max():%Y-%m-%d} (UTC 저장, 패턴은 {tz})",
        f"- 원본 {rep.rows_in:,}행 → 범위 밖 제거 {rep.removed_out_of_range} → {rep.rows_out:,}행",
        f"- 분석 간격 {cfg['preprocess']['resample']}, 같은 공간 장치 평균, {cfg['preprocess']['short_gap_max_minutes']}분 이하 결측만 선형 보간",
        "",
        "## 1. 센서별 기록 주기",
        "같은 공간에 기록 주기가 다른 센서가 섞이면, 5분 칸마다 평균에 들어가는 센서 수가 달라져 값이 흔들린다.",
        "",
        _md(intervals),
        "",
        "## 2. 결측률 (%)",
        "",
        _md(miss),
        "",
        "### 1시간 이상 결측 구간 (긴 순, UTC)",
        "",
        _md(gaps, index=False),
        "",
        "## 3. 시간대별 CO₂ 평균 (ppm, 한국 시간)",
        "",
        "![시간대별 CO2](figures/02_co2_hourly.png)",
        "",
        _md(prof_tbl),
        "",
        "## 4. CO₂ 상승 사건과 반응 시간",
        f"상승 시작 = 15분 사이 +{RISE_PPM:.0f}ppm 이상(직전 15분은 ±{FLAT_PPM:.0f}ppm 이내), 정점까지 +{MIN_RISE_PPM:.0f}ppm 이상인 사건만.",
        "일정표가 없어 '사람이 들어온 시각'을 상승 시작으로 근사했다.",
        "",
        _md(ev_summary, index=False),
        "",
        "## 5. 평소와 어긋난 평일 (평일 평균 하루 모양과의 RMSE)",
        "",
    ]
    for sp, df in odd.items():
        lines += [f"### {sp}", "", _md(df, index=False), ""]
    lines += [
        "## 해석 (직접 작성)",
        "- 결측 원인 추정 (센서 배터리·게이트웨이·수집 중단 중 무엇인지, 시간대가 겹치는지):",
        "- 분석 간격 5분을 고른 근거 (기록 주기 표 참고):",
        "- 평일 패턴과 회의실·실습실 차이:",
        "- CO₂ 반응 시간이 예측 지평(60분)과 경보(30분 전)에 주는 의미:",
        "- 어긋난 날의 이유 (휴일·행사·센서 문제):",
        "",
    ]
    return lines, profiles


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg = load_yaml("default.yaml")
    raw = load_raw(metrics=cfg["metrics"])
    lines, profiles = build(raw, cfg)
    out = ROOT / "reports" / "02_diagnosis.md"
    (out.parent / "figures").mkdir(parents=True, exist_ok=True)
    if profiles:
        _plot_profiles(profiles, out.parent / "figures" / "02_co2_hourly.png")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
