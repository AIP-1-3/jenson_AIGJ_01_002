"""과제 8 심화 (a): 예측 기반 선제 환기 규칙을 지난 1주에 적용했다면 1,000ppm 초과 시간이 얼마나 줄었을지.

    python -m tsinfer.service.control_sim                 # → reports/08b_control.md
    python -m tsinfer.service.control_sim --lead 30 --days 7

규칙 (공간마다 5분 간격으로 판단)
- 제어 없음: 실측 그대로.
- 반응형: 1,000ppm 이상이 되면 환기 시작, 900 아래로 내려가면 멈춤.
- 예측형: 지금 1,000 이상이거나, 예측이 lead(기본 30)분 안에 1,000 이상이면 환기 시작.
  900 아래이고 lead 안 초과 예측도 없으면 멈춤. 예측은 과제 3 GRU(없으면 추세 기준선).
- 완벽 예측(상한): 실제 미래 값을 예측으로 쓴 경우. 예측이 좋아져도 이 이상은 줄지 않는다.

물리 모델 (단순화)
- 환기 없을 때: C[t+1] = 외기 + (C[t] − 외기)·exp(−5/τn) + G[t].  τn 은 데이터의 정점 후 감쇠에서 추정
  (진단서: 절반 회복 60~65분 → τ≈90분). G[t] 는 '사람 발생량'으로, 실측 변화에서 자연 감쇠를 빼서 구한다.
  → 제어 없음을 시뮬레이션하면 실측이 그대로 나온다 (검산).
- 환기 중: 같은 식에서 τ 만 τv(가정, 민감도 표)로 바꾼다. 발생량 G 는 제어와 무관하다고 본다.
- 예측형이 쓰는 예측: 모델은 실측 이력으로 예측한다. 시뮬레이션 값이 실측과 달라진 뒤에는
  '예측 − 실측에서 자연 감쇠만 했을 때' 를 발생 기여분으로 보고, 시뮬레이션 값의 자연 감쇠에 더한다.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml

OUTDOOR_PPM = 420.0
STOP_PPM = 900.0
TAU_NATURAL_DEFAULT = 90.0               # 추정 실패 시 (진단서 절반 회복 60~65분 / ln2)
TAU_VENT = (10.0, 20.0, 30.0, 45.0)      # 환기 중 시상수 가정 (분)
TAU_VENT_MAIN = 20.0


# ---------------------------------------------------------------- 시상수 추정

def estimate_tau(co2: pd.Series, step_min: int, outdoor: float = OUTDOOR_PPM,
                 min_excess: float = 150.0, min_len: int = 6) -> list[float]:
    """정점 뒤 계속 내려가는 구간마다 ln(C − 외기) 기울기로 τ 를 구한다 (사람이 나간 뒤 자연 감쇠)."""
    v = co2.to_numpy(dtype=float)
    taus, i, n = [], 0, len(v)
    while i < n - 1:
        if not (np.isfinite(v[i]) and v[i] - outdoor > min_excess and v[i + 1] < v[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(v[j + 1]) and v[j + 1] < v[j] and v[j + 1] - outdoor > 20:
            j += 1
        if j - i + 1 >= min_len:
            y = np.log(v[i:j + 1] - outdoor)
            slope = np.polyfit(np.arange(j - i + 1) * step_min, y, 1)[0]
            if slope < 0:
                tau = -1.0 / slope
                if 10 <= tau <= 600:
                    taus.append(float(tau))
        i = j + 1
    return taus


# ---------------------------------------------------------------- 예측 (실측 이력 기준, 한꺼번에)

def trend_predictions(co2: pd.Series, horizons: int, window: int = 6, damping: float = 0.97) -> np.ndarray:
    """service.predictors.TrendBaseline 과 같은 식을 시점 전체에 한 번에. (N, horizons)."""
    last = co2.to_numpy(dtype=float)
    slope = (co2 - co2.shift(window)).to_numpy(dtype=float) / window
    acc = np.cumsum(damping ** np.arange(1, horizons + 1))
    return np.maximum(last[:, None] + slope[:, None] * acc[None, :], 400.0)


def oracle_predictions(co2: pd.Series, horizons: int) -> np.ndarray:
    return np.stack([co2.shift(-k).to_numpy(dtype=float) for k in range(1, horizons + 1)], axis=1)


def gru_predictions(wide: pd.DataFrame, at: pd.DatetimeIndex, path: Path | None = None) -> tuple[dict[str, np.ndarray], str]:
    """과제 3 GRU 로 at 시점들의 CO₂ 5~60분 예측. 공간별 (len(at), H), 창에 결측이 있으면 NaN."""
    from tsinfer.models import forecast

    model, spec = forecast.load(path or forecast.MODEL_PATH)
    x, _, info = forecast.make_windows(wide, spec, at=at, with_target=False)
    pred = forecast.decode(forecast.predict_array(model, x), info, spec)["co2"] if len(x) else None
    out = {}
    for sp in spec.spaces:
        arr = np.full((len(at), len(spec.horizons)), np.nan)
        if pred is not None:
            sel = info["space"].to_numpy() == sp
            pos = at.get_indexer(pd.DatetimeIndex(info.loc[sel, "time"]))
            arr[pos[pos >= 0]] = np.maximum(pred[sel][pos >= 0], 400.0)
        out[sp] = arr
    return out, f"GRU(lookback={spec.lookback * spec.step_minutes}m)"


# ---------------------------------------------------------------- 시뮬레이션

def simulate(obs: np.ndarray, a_n: float, a_v: float, policy: str, pred: np.ndarray | None = None,
             lead_steps: int = 6, threshold: float = 1000.0, stop: float = STOP_PPM,
             outdoor: float = OUTDOOR_PPM) -> tuple[np.ndarray, np.ndarray]:
    """시뮬레이션 CO₂, 환기 여부. policy: none | reactive | predictive. 결측 구간에서는 실측으로 다시 시작."""
    n = len(obs)
    gen = np.full(n, np.nan)
    gen[:-1] = obs[1:] - (outdoor + (obs[:-1] - outdoor) * a_n)     # 사람 발생량 (실측 변화 − 자연 감쇠)
    decay_k = a_n ** np.arange(1, lead_steps + 1)
    c_sim, vent = np.full(n, np.nan), np.zeros(n, bool)
    c, on = np.nan, False
    for t in range(n):
        if not np.isfinite(obs[t]):
            c, on = np.nan, False
            continue
        if not np.isfinite(c):
            c = obs[t]
        if policy == "reactive":
            on = c >= threshold or (on and c >= stop)
        elif policy == "predictive":
            exceed = False
            if pred is not None and np.isfinite(pred[t, :lead_steps]).all():
                # 예측의 발생 기여분 = 예측 − (실측이 자연 감쇠만 했을 때) → 시뮬레이션 값에 옮겨 붙인다
                excess = pred[t, :lead_steps] - (outdoor + (obs[t] - outdoor) * decay_k)
                exceed = bool((outdoor + (c - outdoor) * decay_k + excess >= threshold).any())
            on = c >= threshold or exceed or (on and c >= stop)
        c_sim[t], vent[t] = c, on
        if t + 1 < n and np.isfinite(gen[t]):
            c = outdoor + (c - outdoor) * (a_v if on else a_n) + gen[t]
            if on:
                c = max(c, outdoor)
        else:
            c = np.nan
    return c_sim, vent


def _summary(c: np.ndarray, vent: np.ndarray, step_min: int, threshold: float) -> dict[str, float]:
    ok = np.isfinite(c)
    starts = int((vent[1:] & ~vent[:-1]).sum() + (vent[0] if len(vent) else 0))
    return {"exceed_min": int((c[ok] > threshold).sum() * step_min),
            "vent_min": int(vent.sum() * step_min), "vent_starts": starts,
            "peak_ppm": round(float(np.nanmax(c)), 0) if ok.any() else np.nan}


def run(cfg: dict[str, Any], wide: pd.DataFrame, days: int = 7, lead_min: int = 30,
        tau_vents: tuple[float, ...] = TAU_VENT, model_path: Path | None = None, log=print) -> dict[str, Any]:
    from tsinfer.data.split import split_by_time

    step = cfg["forecast"]["step_minutes"]
    H = cfg["forecast"]["horizon_minutes"] // step
    th = float(cfg["thresholds"]["co2_ventilate_ppm"])
    lead = max(1, min(lead_min // step, H))
    _, _, test = split_by_time(wide, cfg)
    if test.empty:
        raise ValueError("시험 기간 데이터가 없습니다 (configs/default.yaml split.valid_end 확인).")
    end = test.index.max()
    idx = test.index[test.index > end - pd.Timedelta(days=days)]
    spaces = [sp for sp in cfg["spaces"] if (sp, "co2") in wide and wide.loc[idx, (sp, "co2")].notna().any()]

    # 자연 감쇠 τn: 평가 1주 이전 데이터에서 (평가 기간을 엿보지 않는다)
    before = wide[wide.index < idx[0]]
    tau_rows, tau_n = [], {}
    for sp in spaces:
        taus = estimate_tau(before[(sp, "co2")], step) if (sp, "co2") in before else []
        tau_n[sp] = float(np.median(taus)) if len(taus) >= 5 else TAU_NATURAL_DEFAULT
        tau_rows.append({"space": sp, "segments": len(taus), "tau_n_min": round(tau_n[sp], 1),
                         "half_life_min": round(tau_n[sp] * np.log(2), 1),
                         "source": "추정" if len(taus) >= 5 else f"기본값 {TAU_NATURAL_DEFAULT:.0f}"})

    t0 = time.perf_counter()
    try:
        preds, model_name = gru_predictions(wide, idx, model_path)
    except Exception as e:   # 모델 파일·torch 가 없으면 추세 기준선 (표에 표시)
        log(f"GRU 를 못 불러와 추세 기준선으로 대체합니다: {e!r}")
        preds, model_name = {}, None
    rows, sens = [], []
    for sp in spaces:
        co2 = wide[(sp, "co2")]
        obs = co2.loc[idx].to_numpy(dtype=float)
        pos = wide.index.get_indexer(idx)
        p_model = preds.get(sp)
        name = model_name
        if p_model is None or not np.isfinite(p_model).any():
            p_model, name = trend_predictions(co2, H)[pos], "추세 기준선(대체)"
        p_oracle = oracle_predictions(co2, H)[pos]
        a_n = float(np.exp(-step / tau_n[sp]))
        for tv in tau_vents:
            a_v = float(np.exp(-step / tv))
            variants = {"제어 없음(실측)": ("none", None), "반응형": ("reactive", None),
                        f"예측형: {name}": ("predictive", p_model), "예측형: 완벽 예측(상한)": ("predictive", p_oracle)}
            for label, (pol, pr) in variants.items():
                c, vent = simulate(obs, a_n, a_v, pol, pr, lead_steps=lead, threshold=th)
                s = _summary(c, vent, step, th)
                rec = {"space": sp, "tau_vent_min": tv, "policy": label, **s}
                if pol == "none":
                    rec["replay_err_ppm"] = float(np.nanmax(np.abs(c - obs))) if np.isfinite(c).any() else 0.0
                sens.append(rec)
    res = pd.DataFrame(sens)
    log(f"시뮬레이션 {time.perf_counter() - t0:.1f}s")
    return {"index": idx, "spaces": spaces, "tau": pd.DataFrame(tau_rows), "results": res,
            "model": model_name or "추세 기준선(대체: GRU 모델 없음)", "lead_min": lead * step,
            "threshold": th, "step": step}


def _with_reduction(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["policy"] == "제어 없음(실측)"].set_index(["space", "tau_vent_min"])["exceed_min"]
    key = list(zip(df["space"], df["tau_vent_min"]))
    b = np.array([base.get(k, np.nan) for k in key], dtype=float)
    df = df.copy()
    df["감소_%"] = np.where(b > 0, np.round((1 - df["exceed_min"] / b) * 100, 1), np.nan)
    return df


def report(out: dict[str, Any], days: int, elapsed: float) -> str:
    res = _with_reduction(out["results"])
    main_t = res[res["tau_vent_min"] == TAU_VENT_MAIN].drop(columns=["tau_vent_min", "replay_err_ppm"], errors="ignore")
    total = res.groupby(["tau_vent_min", "policy"], sort=False)[["exceed_min", "vent_min", "vent_starts"]].sum().reset_index()
    base = total[total["policy"] == "제어 없음(실측)"].set_index("tau_vent_min")["exceed_min"]
    total["감소_%"] = [round((1 - e / base[t]) * 100, 1) if base.get(t, 0) else np.nan
                     for t, e in zip(total["tau_vent_min"], total["exceed_min"])]
    sens = total.pivot_table(index="policy", columns="tau_vent_min", values="exceed_min", sort=False)
    sens.columns = [f"τv={c:.0f}분" for c in sens.columns]
    replay = res["replay_err_ppm"].max() if "replay_err_ppm" in res else np.nan
    idx = out["index"]
    lines = [
        "# 과제 8 심화 (a). 예측 기반 선제 환기 — 1,000ppm 초과 시간 추정",
        "",
        "## 측정 조건",
        f"- 기간: 시험 기간 마지막 {days}일 {idx.min():%Y-%m-%d %H:%M} ~ {idx.max():%Y-%m-%d %H:%M} (UTC), "
        f"{out['step']}분 간격, 공간 {', '.join(out['spaces'])}",
        f"- 예측: {out['model']} — 실측 이력으로 한 번에 예측해 쓰고, 시뮬레이션 값에는 발생 기여분만 옮겨 붙인다",
        f"- 규칙: 반응형 = {out['threshold']:.0f} 이상이면 환기, {STOP_PPM:.0f} 미만이면 멈춤 / "
        f"예측형 = 지금 {out['threshold']:.0f} 이상이거나 {out['lead_min']}분 안 초과 예측이면 환기, "
        f"{STOP_PPM:.0f} 미만이고 초과 예측 없으면 멈춤",
        f"- 물리 모델: 외기 {OUTDOOR_PPM:.0f}ppm 쪽 지수 감쇠. 자연 τn 은 평가 기간 이전 데이터의 정점 후 감쇠에서 추정, "
        f"환기 중 τv 는 가정값 (본표 {TAU_VENT_MAIN:.0f}분, 민감도 {', '.join(f'{t:.0f}' for t in TAU_VENT)}분)",
        f"- 사람 발생량 = 실측 변화 − 자연 감쇠 (제어와 무관하다고 가정). 검산: 제어 없음 재현 최대 오차 {replay:.2g}ppm",
        f"- 초과 시간 = 5분 칸 값 > {out['threshold']:.0f} 인 칸 수 × 5분 (결측 칸은 세지 않음). 실행 {elapsed:.0f}s",
        "",
        "## 자연 감쇠 시상수 τn",
        "",
        out["tau"].to_markdown(index=False),
        "",
        f"## 결과 (τv = {TAU_VENT_MAIN:.0f}분): 공간별 초과 시간(분), 환기 시간(분), 환기 시작 횟수",
        "",
        main_t.to_markdown(index=False),
        "",
        f"## 공간 합계 (τv = {TAU_VENT_MAIN:.0f}분)",
        "",
        total[total["tau_vent_min"] == TAU_VENT_MAIN].drop(columns="tau_vent_min").to_markdown(index=False),
        "",
        "## 민감도: 환기 시상수 τv 에 따른 공간 합계 초과 시간(분)",
        "",
        sens.to_markdown(),
        "",
        "## 가정과 한계",
        "- 공간을 한 덩어리 공기(완전 혼합)로 본다. 실제로는 센서 위치·문 열림에 따라 다르다.",
        "- 환기 효과(τv)는 측정값이 아니라 가정이다. 실제 환기 장치의 풍량·공간 부피로 τv = 부피/풍량 을 구하면 바꿔 넣는다.",
        "- 발생량을 실측에서 역산하므로, 실측 기간에 이미 창문을 연 효과(음의 발생량)가 섞여 있다.",
        "- 사람이 환기 때문에 행동을 바꾸지 않는다고 본다. 예측은 실측 이력으로 만들어, 제어 후 이력과 다르다.",
        "- 환기 시간(분)은 에너지 비용의 대리 지표. 예측형은 일찍 켜는 만큼 환기 시간이 늘 수 있다.",
        "",
        "## 해석 (직접 작성)",
        "- 예측형이 반응형보다 줄인 초과 시간과, 늘어난 환기 시간(비용):",
        "- 완벽 예측(상한)과의 차이 = 예측 오차 때문에 놓친 몫:",
        "- τv 가정에 따라 결론이 바뀌는지:",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--lead", type=int, default=30, help="예측형이 미리 보는 시간(분)")
    args = ap.parse_args()

    from tsinfer.data.preprocess import clean, load_raw, to_wide

    t0 = time.perf_counter()
    cfg = load_yaml("default.yaml")
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    out = run(cfg, wide, days=args.days, lead_min=args.lead)
    text = report(out, args.days, time.perf_counter() - t0)
    path = ROOT / "reports" / "08b_control.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    out["results"].to_csv(ROOT / "reports" / "08b_control.csv", index=False)
    print(text)
    print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
