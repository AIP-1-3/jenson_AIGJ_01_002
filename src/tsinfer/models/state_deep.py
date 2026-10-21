"""과제 5 심화: 학습하지 않은 공간에서의 상태 추정 + 시간표 대조 규칙.

    python -m tsinfer.models.state_deep       # → reports/05b_state_deep.md (모델 파일은 저장하지 않음)

(a) 공간 하나 빼고 학습 (leave-one-space-out)
    나머지 공간의 학습 기간으로 학습 → 빠진 공간의 시험 기간을 채점한다.
    같은 공간까지 넣어 학습한 모델(과제 5 와 같은 방식)·규칙과 같은 시험 행에서 비교한다.
    입력·정답·임계값 정하는 법은 state.py 와 같다 (사용 여부 = 센서만·0.5, 환기 경보 = 교차 예측 재현율 운영점).
(b) 시간표 대조 규칙 (회의실 노쇼·무예약 사용의 대체)
    예약표·직접 관찰 기록이 없어 회의실 노쇼는 만들 수 없다. 대신 시간표가 확정된 실습실에
    '수업인데 비어 있음'(수업 칸 시작 후 15분 내내 비어 있다고 추정)과
    '수업 아닌데 사용'(수업 밖에서 15분 이상 계속 사용 중이라고 추정)을 같은 틀로 세어 본다.
    추정기는 센서 규칙과, 그 공간을 빼고 학습한 hgb (그 공간의 시간표 정답을 한 번도 보지 않은 모델) 두 가지.
(c) 환기 경보 — 과제 3 예측 모델로 판단 vs 분류기 따로
    저장된 GRU(models/forecast_gru.pt)로 각 시점의 5~30분 CO₂ 예측 최댓값을 구해 '≥ 임계값이면 경보'(예측 기반),
    그리고 hgb 입력에 GRU 예측 최댓값·규칙 신호를 더한 모델(결합 hgb)을 규칙·기존 hgb 와 같은 시험 시점에서 비교한다.
    임계값·모델은 학습+검증 기간에서만 정하고 시험 기간은 채점에만 쓴다. GRU 가 없으면 그 행만 '없음'.
(d) 직접 관찰 대용 — 실습실 AM107 PIR
    수집기 SQLite(읽기 전용)의 pir_activity 가 있는 기간만, 5분 칸별 PIR 활동 vs 시간표 정답 vs 센서 추정을 비교한다.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import state as st

SLOT_MIN = 60          # 시간표를 이 길이의 칸(예약 단위처럼)으로 잘라 칸마다 판정
NOSHOW_MIN = 15        # 칸 시작 후 이만큼 내내 비어 있으면 '수업인데 비어 있음'
UNSCHEDULED_MIN = 15   # 수업 밖에서 이만큼 이상 계속 사용 중이면 '수업 아닌데 사용'
TIMETABLE_SPACE = "실습실"


# ---------------------------------------------------------------- (a) 공간 하나 빼고 학습

def _parts(df: pd.DataFrame, wide: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.Series]:
    from tsinfer.data.split import split_by_time

    tr_idx, va_idx, te_idx = (s.index for s in split_by_time(wide, cfg))
    return {"train": df["time"].isin(tr_idx), "valid": df["time"].isin(va_idx), "test": df["time"].isin(te_idx)}


def fit_model(task: str, fit_df: pd.DataFrame, fx: list[str], name: str, seed: int, ycol: str,
              folds: int = st.CV_FOLDS) -> tuple[Any, float]:
    """state.run 과 같은 학습·임계값 규칙 (사용 여부 0.5, 환기 경보 = 교차 예측 재현율 운영점)."""
    thr = 0.5
    if task == "ventilation":
        fit_df = fit_df.sort_values("time")      # 교차 예측 구간이 '시간순'이 되도록 (공간이 섞여 있음)
        y_fit = fit_df[ycol].to_numpy().astype(int)
        p_oof = st.oof_proba(lambda: st.make_models(seed)[name], fit_df[fx], y_fit, folds)
        okp = ~np.isnan(p_oof)
        if y_fit[okp].sum() > 0:
            thr = st.pick_threshold(y_fit[okp], p_oof[okp])
    model = st.make_models(seed)[name]
    model.fit(fit_df[fx], fit_df[ycol].astype(int))
    return model, thr


def loso(cfg: dict[str, Any], wide: pd.DataFrame, sched: dict[str, Any], observations=None,
         df: pd.DataFrame | None = None, models=("logreg", "hgb"), folds: int = st.CV_FOLDS,
         tasks=("occupied", "ventilation")) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """과제별 표: 행 = (시험 공간, 방법). 방법 = 규칙 / <모델>(같은 공간 포함) / <모델>(이 공간 빼고).

    '같은 공간 포함' 모델은 모든 공간으로 한 번만 학습해 공간별로 나눠 채점한다 (학습 횟수 절약).
    둘째 반환값: 공간별 '그 공간을 빼고 학습한 hgb 사용 여부 모델' (시간표 대조 규칙에서 쓴다)."""
    th = cfg["thresholds"]
    df = st.dataset(wide, cfg, sched, observations) if df is None else df
    part = _parts(df, wide, cfg)
    feats_all = [c for c in df.columns if c not in st.FEATURES_EXCLUDE]
    feats = {"occupied": [c for c in feats_all if c not in st.TIME_FEATURES], "ventilation": feats_all}
    seed = cfg.get("model", {}).get("seed", 0)
    spaces = list(df["space"].unique())
    tables, held_models = {}, {}
    for task, ycol in (("occupied", "y_occ"), ("ventilation", "y_vent")):
        if task not in tasks:
            continue
        ok = df[ycol].notna() & df["co2"].notna()
        fit_mask = ok & (part["train"] | part["valid"]) if task == "ventilation" else ok & part["train"]
        fx = feats[task]
        shared = {}
        for name in models:
            fit_df = df[fit_mask]
            shared[name] = fit_model(task, fit_df, fx, name, seed, ycol, folds) if fit_df[ycol].nunique() > 1 else None
        rows = []
        for sp in spaces:
            te = df[ok & part["test"] & (df["space"] == sp)]
            if te.empty:
                continue
            y_te = te[ycol].to_numpy().astype(int)
            rule = st.rule_occupied(te) if task == "occupied" else st.rule_vent(
                te, th["co2_ventilate_ppm"], th["ventilation_alert_within_minutes"])
            rows.append({"space": sp, "method": "규칙", **st.scores(y_te, rule.astype(int))})
            for name in models:
                fit_df = df[fit_mask & (df["space"] != sp)]
                held = fit_model(task, fit_df, fx, name, seed, ycol, folds) if fit_df[ycol].nunique() > 1 else None
                for label, mt in (("같은 공간 포함", shared[name]), ("이 공간 빼고", held)):
                    if mt is None:
                        rows.append({"space": sp, "method": f"{name} ({label})", "note": "학습 정답이 한 종류"})
                        continue
                    model, thr = mt
                    pred = (model.predict_proba(te[fx])[:, 1] >= thr).astype(int)
                    rows.append({"space": sp, "method": f"{name} ({label})", **st.scores(y_te, pred),
                                 "threshold": round(thr, 3)})
                if task == "occupied" and name == "hgb" and held is not None:
                    held_models[sp] = held[0]
        if rows:
            tables[task] = pd.DataFrame(rows)
    return tables, held_models


def loso_summary(table: pd.DataFrame) -> pd.DataFrame:
    """방법별 공간 평균 macro-F1·빈방→사용중 오류율, 그리고 '이 공간 빼고'가 '같은 공간 포함'보다 잃은 macro-F1."""
    t = table.dropna(subset=["macro_f1"])
    g = t.groupby("method")[["macro_f1", "empty_as_occupied_%", "recall", "precision"]].mean().round(3)
    for name in ("logreg", "hgb"):
        a, b = f"{name} (같은 공간 포함)", f"{name} (이 공간 빼고)"
        if a in g.index and b in g.index:
            g.loc[b, "macro_f1_drop"] = round(g.loc[a, "macro_f1"] - g.loc[b, "macro_f1"], 3)
    return g


# ---------------------------------------------------------------- (b) 시간표 대조 규칙

def runs(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """True 가 이어진 구간 (시작, 끝(마지막 칸), 칸 수). 시점 간격이 끊기면 다른 구간으로 본다."""
    out = []
    if flag.empty:
        return out
    step = flag.index.to_series().diff().median() if len(flag) > 1 else pd.Timedelta("5min")
    v = flag.to_numpy().astype(bool)
    t = flag.index
    start = None
    for i in range(len(v)):
        cont = start is not None and t[i] - t[i - 1] <= step
        if v[i] and start is None:
            start = i
        elif start is not None and (not v[i] or not cont):
            out.append((t[start], t[i - 1], i - start))
            start = i if v[i] else None
    if start is not None:
        out.append((t[start], t[len(v) - 1], len(v) - start))
    return out


def schedule_mismatch(est: pd.Series, sched_occ: pd.Series, tz: str, step_min: int = 5, slot_min: int = SLOT_MIN,
                      noshow_min: int = NOSHOW_MIN, unscheduled_min: int = UNSCHEDULED_MIN,
                      holidays=()) -> pd.DataFrame:
    """est·sched_occ: 같은 시점 색인의 0/1 (est 결측 = 판정 보류).

    '수업인데 비어 있음': 수업 칸(시간표 구간을 slot_min 씩 자른 칸)의 첫 noshow_min 동안 추정이 전부 0.
    '수업 아닌데 사용'  : 시간표 밖에서 추정 1 이 unscheduled_min 이상 이어진 구간.
    반환: kind, start(한국 시간), end, minutes, holiday."""
    need_ns = noshow_min // step_min
    need_un = unscheduled_min // step_min
    hol = set(str(d) for d in holidays)
    rows = []
    sched_on = sched_occ.fillna(0).astype(bool)
    for s, e, n in runs(sched_on):
        blk = pd.date_range(s, e, freq=f"{step_min}min")
        for k in range(0, len(blk), slot_min // step_min):
            head = est.reindex(blk[k:k + need_ns])
            if len(head) < need_ns or head.isna().any():
                continue
            if (head == 0).all():
                st_local = blk[k].tz_convert(tz)
                rows.append({"kind": "수업인데 비어 있음", "start": st_local, "end": st_local + pd.Timedelta(minutes=noshow_min),
                             "minutes": noshow_min, "holiday": st_local.strftime("%Y-%m-%d") in hol})
    used_off = (est.fillna(0) == 1) & ~sched_on
    for s, e, n in runs(used_off):
        if n >= need_un:
            s_l = s.tz_convert(tz)
            rows.append({"kind": "수업 아닌데 사용", "start": s_l, "end": e.tz_convert(tz) + pd.Timedelta(minutes=step_min),
                         "minutes": n * step_min, "holiday": s_l.strftime("%Y-%m-%d") in hol})
    return pd.DataFrame(rows, columns=["kind", "start", "end", "minutes", "holiday"])


def mismatch_report(df: pd.DataFrame, wide: pd.DataFrame, cfg: dict[str, Any], space: str,
                    held_model=None) -> dict[str, pd.DataFrame]:
    """시험 기간 그 공간에서 추정기별 사건 목록."""
    tz = cfg.get("timezone", "Asia/Seoul")
    step = cfg["forecast"]["step_minutes"]
    part = _parts(df, wide, cfg)
    te = df[part["test"] & (df["space"] == space)].set_index("time").sort_index()
    if te.empty:
        return {}
    sched_occ = te["y_occ"]
    ok = te["co2"].notna()
    feats = [c for c in df.columns if c not in st.FEATURES_EXCLUDE and c not in st.TIME_FEATURES]
    ests = {"규칙": pd.Series(np.where(ok, st.rule_occupied(te), np.nan), index=te.index)}
    if held_model is not None:
        ests[f"hgb ({space} 빼고 학습)"] = pd.Series(
            np.where(ok, (held_model.predict_proba(te[feats])[:, 1] >= 0.5), np.nan), index=te.index)
    return {name: schedule_mismatch(e, sched_occ, tz, step, holidays=cfg.get("holidays", []))
            for name, e in ests.items()}


def mismatch_counts(events: dict[str, pd.DataFrame], days: float) -> pd.DataFrame:
    rows = []
    for name, ev in events.items():
        for kind in ("수업인데 비어 있음", "수업 아닌데 사용"):
            e = ev[ev["kind"] == kind]
            rows.append({"estimator": name, "kind": kind, "events": len(e), "per_day": round(len(e) / max(days, 1e-9), 2),
                         "on_holiday": int(e["holiday"].sum()), "minutes_total": int(e["minutes"].sum())})
    return pd.DataFrame(rows)


def mismatch_by_hour(events: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for name, ev in events.items():
        if len(ev):
            parts.append(ev.assign(estimator=name, hour=pd.to_datetime(ev["start"]).map(lambda t: t.hour)))
    if not parts:
        return pd.DataFrame()
    d = pd.concat(parts)
    return d.pivot_table(index=["estimator", "kind"], columns="hour", values="minutes", aggfunc="size", fill_value=0)


# ---------------------------------------------------------------- (c) 환기 경보: 예측 기반 vs 분류기

RULE_PPM = 850                 # 규칙 신호: 현재 CO₂ 가 이 값 이상 (결합 hgb 입력·참고 행)
COMBINED_EXTRA = ("gru_max30", "gru_rise30", "co2_ge850", "rule_vent")
VENT_COLS = ["n", "pos", "alarms", "macro_f1", "recall", "precision", "false_alarm_%", "fp", "fn", "threshold", "pass"]


def gru_forecast_max(wide: pd.DataFrame, cfg: dict[str, Any], model_path: Path | None = None,
                     ) -> tuple[pd.DataFrame | None, str]:
    """저장된 GRU 로 시점·공간별 '5~(경보 시간)분 CO₂ 예측 최댓값'. 실패하면 (None, 사유).

    forecast.load → make_windows(with_target=False) → predict_array → decode 경로를 그대로 쓴다.
    공간마다 따로 창을 만들어 메모리를 줄인다 (CPU 추론, GPU 불필요)."""
    try:
        from tsinfer.models import forecast as fc

        path = Path(model_path) if model_path is not None else fc.MODEL_PATH
        if not path.exists():
            return None, f"GRU 파일 없음 ({path.name})"
        import torch

        torch.set_num_threads(min(int(cfg.get("model", {}).get("threads", 2)), 3))
        model, spec = fc.load(path)
    except Exception as e:                          # torch 미설치·파일 손상·구조 불일치
        return None, f"GRU 불러오기 실패 ({type(e).__name__}: {e})"
    within = cfg["thresholds"]["ventilation_alert_within_minutes"]
    hs = [k for k, h in enumerate(spec.horizons) if h * spec.step_minutes <= within]
    if not hs:
        return None, f"GRU 지평에 {within}분 이내 칸이 없음"
    parts = []
    try:
        for sp in spec.spaces:
            x, _, info = fc.make_windows(wide, spec, with_target=False, spaces=[sp])
            if not len(x):
                continue
            co2 = fc.decode(fc.predict_array(model, x), info, spec)["co2"]
            parts.append(pd.DataFrame({"time": info["time"].to_numpy(), "space": sp,
                                       "gru_max30": co2[:, hs].max(axis=1)}))
            del x
    except Exception as e:
        return None, f"GRU 예측 실패 ({type(e).__name__}: {e})"
    if not parts:
        return None, "GRU 입력 창을 만들 수 있는 시점이 없음"
    return pd.concat(parts, ignore_index=True), f"GRU {path.name} (지평 {within}분 이내 {len(hs)}칸 최댓값)"


def sweep(y: np.ndarray, s: np.ndarray, cands: np.ndarray) -> pd.DataFrame:
    """임계값 후보별 재현율·정밀도·macro-F1 ('s ≥ 임계값이면 경보')."""
    y = y.astype(bool)
    rows = []
    for t in cands:
        p = s >= t
        tp = int((p & y).sum()); fp = int((p & ~y).sum())
        fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
        f_pos = 2 * tp / max(2 * tp + fp + fn, 1)
        f_neg = 2 * tn / max(2 * tn + fn + fp, 1)
        rows.append({"threshold": float(t), "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
                     "macro_f1": (f_pos + f_neg) / 2})
    return pd.DataFrame(rows)


def pick_two(y: np.ndarray, s: np.ndarray, recall_target: float = st.RECALL_TARGET) -> tuple[float, float]:
    """(macro-F1 최대 임계값, 재현율 ≥ 목표 중 macro-F1 최대 임계값). 목표를 못 채우면 재현율이 가장 큰 값."""
    s = np.asarray(s, float)
    cands = np.unique(np.quantile(s, np.linspace(0.3, 0.9995, 400)))
    sw = sweep(y, s, cands)
    thr_f1 = float(sw.loc[sw["macro_f1"].idxmax(), "threshold"])
    ok = sw[sw["recall"] >= recall_target]
    thr_rec = float(ok.loc[ok["macro_f1"].idxmax(), "threshold"]) if len(ok) else float(
        sw.sort_values(["recall", "macro_f1"]).iloc[-1]["threshold"])
    return thr_f1, thr_rec


def _vent_row(method: str, y: np.ndarray, pred: np.ndarray, thr=None, rule_f1: float | None = None) -> dict:
    s = st.scores(y, pred.astype(int))
    row = {"method": method, "n": s["n"], "pos": int(y.sum()), "alarms": int(pred.sum()),
           "macro_f1": s["macro_f1"], "recall": s["recall"], "precision": s["precision"],
           "false_alarm_%": s["empty_as_occupied_%"], "fp": s["fp"], "fn": s["fn"],
           "threshold": None if thr is None else round(float(thr), 3)}
    if rule_f1 is not None:
        row["pass"] = bool(s["recall"] >= st.RECALL_TARGET and s["macro_f1"] > rule_f1)
    return row


def forecast_alarm(cfg: dict[str, Any], wide: pd.DataFrame, df: pd.DataFrame, model_path: Path | None = None,
                   folds: int = st.CV_FOLDS) -> dict[str, Any]:
    """규칙 / 기존 hgb / 예측 기반 / 결합 hgb 를 같은 시험 시점에서 채점.

    양성 정의는 state.py 그대로 (이미 임계값을 넘은 시점 제외). GRU 가 있으면 시험 행을 'GRU 예측이 있는 시점'으로
    맞춰 모든 행을 같은 시점에서 비교한다. 임계값·학습은 학습+검증 기간만 쓴다."""
    th = cfg["thresholds"]
    ppm, within = th["co2_ventilate_ppm"], th["ventilation_alert_within_minutes"]
    seed = cfg.get("model", {}).get("seed", 0)
    part = _parts(df, wide, cfg)
    g, gnote = gru_forecast_max(wide, cfg, model_path)
    d = df.copy()
    if g is not None:
        d = d.merge(g, on=["time", "space"], how="left")       # 왼쪽 순서 유지 → part 마스크와 행이 맞는다
    else:
        d["gru_max30"] = np.nan
    d["gru_rise30"] = d["gru_max30"] - d["co2"]
    d["co2_ge850"] = (d["co2"] >= RULE_PPM).astype(float)
    d["rule_vent"] = st.rule_vent(d, ppm, within).astype(float)
    d.index = df.index
    ok = d["y_vent"].notna() & d["co2"].notna()
    has_g = d["gru_max30"].notna()
    fit_mask = ok & (part["train"] | part["valid"])
    te_mask = ok & part["test"] & (has_g if g is not None else True)
    fit, te = d[fit_mask], d[te_mask]
    info = {"gru": gnote, "fit_rows": int(len(fit)), "fit_pos": int(fit["y_vent"].sum()),
            "test_rows": int(len(te)), "test_pos": int(te["y_vent"].sum()) if len(te) else 0,
            "test_rows_state": int((ok & part["test"]).sum())}
    if te.empty or fit["y_vent"].nunique() < 2:
        return {"table": pd.DataFrame(), "info": {**info, "skip": "시험 행이 없거나 학습 정답이 한 종류"}}
    y_te = te["y_vent"].to_numpy().astype(int)
    base = [c for c in df.columns if c not in st.FEATURES_EXCLUDE]
    comb = base + list(COMBINED_EXTRA)

    rule = st.rule_vent(te, ppm, within)
    rule_f1 = st.scores(y_te, rule.astype(int))["macro_f1"]
    rows = [_vent_row("규칙 (state.py: 30분 기울기 연장)", y_te, rule, rule_f1=rule_f1),
            _vent_row(f"규칙 (현재 CO₂ ≥{RULE_PPM}, 참고)", y_te, (te["co2"] >= RULE_PPM).to_numpy(), rule_f1=rule_f1)]

    # 기존 hgb: state.py 와 같은 절차 (학습+검증 시간순 교차 예측으로 임계값, 학습+검증으로 다시 학습)
    y_fit = fit["y_vent"].to_numpy().astype(int)
    p_oof = st.oof_proba(lambda: st.make_models(seed)["hgb"], fit[base], y_fit, folds)
    okp = ~np.isnan(p_oof)
    thr_rec, thr_f1 = st.pick_threshold(y_fit[okp], p_oof[okp]), st.pick_threshold_f1(y_fit[okp], p_oof[okp])
    p_te = st.make_models(seed)["hgb"].fit(fit[base], y_fit).predict_proba(te[base])[:, 1]
    rows += [_vent_row("기존 hgb (재현율 운영점)", y_te, p_te >= thr_rec, thr_rec, rule_f1),
             _vent_row("기존 hgb (F1 최대 운영점)", y_te, p_te >= thr_f1, thr_f1, rule_f1)]

    if g is None:
        for m in ("예측 기반 (F1 최대 운영점)", "예측 기반 (재현율 운영점)",
                  "결합 hgb (F1 최대 운영점)", "결합 hgb (재현율 운영점)"):
            rows.append({"method": m, "note": f"없음 — {gnote}"})
    else:
        fit_g = fit[fit["gru_max30"].notna()]
        yg = fit_g["y_vent"].to_numpy().astype(int)
        # 예측 기반: 'GRU 5~30분 예측 최댓값 ≥ 임계값(ppm)이면 경보'. 임계값은 학습+검증 기간에서만 고른다
        t_f1, t_rec = pick_two(yg, fit_g["gru_max30"].to_numpy())
        s_te = te["gru_max30"].to_numpy()
        rows += [_vent_row("예측 기반 (F1 최대 운영점)", y_te, s_te >= t_f1, t_f1, rule_f1),
                 _vent_row("예측 기반 (재현율 운영점)", y_te, s_te >= t_rec, t_rec, rule_f1)]
        # 결합 hgb: 기존 입력 + GRU 예측 최댓값·상승폭 + 규칙 신호. 기존 hgb 와 같은 교차 예측 절차
        p_oof = st.oof_proba(lambda: st.make_models(seed)["hgb"], fit_g[comb], yg, folds)
        okp = ~np.isnan(p_oof)
        c_f1, c_rec = pick_two(yg[okp], p_oof[okp])
        p_te = st.make_models(seed)["hgb"].fit(fit_g[comb], yg).predict_proba(te[comb])[:, 1]
        rows += [_vent_row("결합 hgb (F1 최대 운영점)", y_te, p_te >= c_f1, c_f1, rule_f1),
                 _vent_row("결합 hgb (재현율 운영점)", y_te, p_te >= c_rec, c_rec, rule_f1)]
    return {"table": pd.DataFrame(rows).set_index("method"), "info": info}


def forecast_alarm_summary(t: pd.DataFrame) -> list[str]:
    """숫자 비교만 (해석은 직접 작성)."""
    if t.empty or "macro_f1" not in t:
        return []
    f = t["macro_f1"]
    rule = f.get("규칙 (state.py: 30분 기울기 연장)")

    def v(m: str) -> str:
        x = f.get(m)
        return "없음" if x is None or pd.isna(x) else f"{x:.3f}"

    out = [f"- macro-F1 (F1 최대 운영점): 규칙 {v('규칙 (state.py: 30분 기울기 연장)')} / 기존 hgb "
           f"{v('기존 hgb (F1 최대 운영점)')} / 예측 기반 {v('예측 기반 (F1 최대 운영점)')} / "
           f"결합 hgb {v('결합 hgb (F1 최대 운영점)')}"]
    a, b = f.get("예측 기반 (F1 최대 운영점)"), f.get("결합 hgb (F1 최대 운영점)")
    if a is not None and b is not None and pd.notna(a) and pd.notna(b):
        rec = t["recall"]
        out.append(f"- 예측 기반 − 결합 hgb: macro-F1 {a - b:+.3f} (F1 운영점), "
                   f"{f['예측 기반 (재현율 운영점)'] - f['결합 hgb (재현율 운영점)']:+.3f} (재현율 운영점, 재현율 "
                   f"{rec['예측 기반 (재현율 운영점)']:.3f} vs {rec['결합 hgb (재현율 운영점)']:.3f})")
    passed = [m for m in t.index if "pass" in t and pd.notna(t.loc[m, "pass"]) and bool(t.loc[m, "pass"])]
    out.append(f"- 확인 기준(재현율 ≥{st.RECALL_TARGET:.0%} 이고 규칙 macro-F1 {rule:.3f} 초과) 만족: "
               + (", ".join(f"{m} ({f[m]:.3f})" for m in passed) if passed else "없음"))
    return out


def forecast_alarm_lines(res: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    th = cfg["thresholds"]
    info = res["info"]
    lines = ["## 4. 환기 경보 — 예측 모델로 판단 vs 분류기 따로", "",
             f"- 양성: 지금 <{th['co2_ventilate_ppm']}ppm 이고 {th['ventilation_alert_within_minutes']}분 안에 "
             f"≥{th['co2_ventilate_ppm']}ppm (state.py 그대로, 이미 넘은 시점 제외)",
             f"- 예측: {info['gru']}",
             "- 예측 기반: 'GRU 5~30분 CO₂ 예측 최댓값 ≥ 임계값(ppm)이면 경보'. 임계값은 학습+검증 기간 시점에서 "
             "macro-F1 최대 / 재현율 ≥85% 중 macro-F1 최대",
             f"- 결합 hgb: 기존 hgb 입력 + GRU 예측 최댓값·(예측 최댓값−현재 CO₂)·현재 CO₂ ≥{RULE_PPM} 여부·"
             "규칙(기울기 연장) 경보 여부. 기존 hgb 처럼 학습+검증 기간 시간순 교차 예측으로 임계값, "
             "학습+검증으로 다시 학습 (GRU 예측이 있는 시점만)",
             "- 기존 hgb: state.py 와 같은 절차·입력·임계값 규칙 (재현율 운영점 = 재현율 ≥85% 인 가장 높은 확률값)",
             f"- 시험 시점: {info['test_rows']}개 (양성 {info['test_pos']}) — 모든 행이 같은 시점. "
             f"05_state.md 의 시험 시점 {info['test_rows_state']}개 중 GRU 입력 창(결측 없는 직전 구간)이 있는 시점만. "
             f"학습+검증 {info['fit_rows']}개 (양성 {info['fit_pos']})",
             "- 열: pos = 정답 양성 수, alarms = 경보 수, false_alarm_% = 음성 중 경보 비율 "
             "(05_state.md 의 empty_as_occupied_% 와 같은 계산), pass = 재현율 ≥85% 이고 규칙보다 macro-F1 높음",
             ""]
    t = res["table"]
    if t.empty:
        return lines + [f"({info.get('skip', '건너뜀')})", ""]
    lines += [t.reindex(columns=[c for c in VENT_COLS + ["note"] if c in t.columns]).to_markdown(), "",
              "### 자동 요약 (숫자 비교)", "", *forecast_alarm_summary(t), "",
              "### 해석 (직접 작성)",
              "- 과제 3 예측 모델로 판단하는 것과 분류기를 따로 두는 것 중 무엇이 나은가:",
              "- 서비스에 넣는다면 어느 운영점·방식으로:", ""]
    return lines


# ---------------------------------------------------------------- (d) 직접 관찰 대용: PIR

PIR_METRIC = "pir_activity"
PIR_SINCE = "2026-10-20"       # 수집기에 AM107 PIR 이 들어오기 시작한 날 (한국 시간)
PIR_MAX_GAP_MIN = 30           # 보고 간격이 이보다 길면 그 사이는 '모름'


def _load_db(path: Path, sql: str, params: tuple) -> pd.DataFrame:
    # 수집기가 쓰는 중에도 읽기만 하도록 읽기 전용 (coverage.load 와 같은 방식)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def pir_bins(pir: pd.DataFrame, step_min: int = 5) -> pd.Series:
    """PIR 기록(ts, value) → 5분 칸별 활동 0/1. 한 기록은 '직전 기록 이후 ~ 이번 기록' 동안의 활동으로 보고
    그 사이 칸에 펼친다 (보고 간격이 PIR_MAX_GAP_MIN 보다 길면 보고 간격 중앙값만큼만). 칸이 겹치면 활동 우선."""
    p = pir.sort_values("ts")
    ts = pd.to_datetime(p["ts"], utc=True, format="ISO8601").reset_index(drop=True)
    val = (p["value"].to_numpy() > 0).astype(float)
    dt = ts.diff()
    typical = dt.median() if len(ts) > 1 and pd.notna(dt.median()) else pd.Timedelta(minutes=step_min)
    typical = min(typical, pd.Timedelta(minutes=PIR_MAX_GAP_MIN))
    freq = f"{step_min}min"
    out: dict[pd.Timestamp, float] = {}
    for i, t in enumerate(ts):
        gap = dt.iloc[i]
        start = ts.iloc[i - 1] if i > 0 and gap <= pd.Timedelta(minutes=PIR_MAX_GAP_MIN) else t - typical
        last = (t - pd.Timedelta(microseconds=1)).floor(freq)
        for b in pd.date_range(min(start.floor(freq), last), last, freq=freq):
            out[b] = max(out.get(b, 0.0), val[i])
    return pd.Series(out, dtype=float).sort_index()


def occupancy_estimator(df: pd.DataFrame, wide: pd.DataFrame, cfg: dict[str, Any],
                        state_model_path: Path | None = None) -> tuple[Any, list[str], str]:
    """저장된 사용 여부 hgb(models/state.joblib)를 쓰고, 없거나 못 읽으면 state.py 와 같은 방식으로 새로 학습."""
    path = Path(state_model_path) if state_model_path is not None else ROOT / "models" / "state.joblib"
    if path.exists():
        try:
            import joblib

            obj = joblib.load(path)
            return obj["models"]["occupied_hgb"], list(obj["features"]["occupied"]), f"hgb (저장된 {path.name})"
        except Exception:
            pass
    part = _parts(df, wide, cfg)
    fx = [c for c in df.columns if c not in st.FEATURES_EXCLUDE and c not in st.TIME_FEATURES]
    tr = df[df["y_occ"].notna() & df["co2"].notna() & part["train"]]
    if tr["y_occ"].nunique() < 2:
        return None, fx, "hgb (학습 정답이 한 종류라 없음)"
    model = st.make_models(cfg.get("model", {}).get("seed", 0))["hgb"].fit(tr[fx], tr["y_occ"].astype(int))
    return model, fx, "hgb (새로 학습: 학습 기간, state.py 와 같은 방식)"


def pir_compare(cfg: dict[str, Any], sched: dict[str, Any], wide: pd.DataFrame, df: pd.DataFrame,
                observations=None, db: Path | None = None, space: str = TIMETABLE_SPACE,
                since: str | None = PIR_SINCE, state_model_path: Path | None = None) -> dict[str, Any]:
    """5분 칸별 PIR 활동 vs 시간표 정답 vs 센서 추정. 데이터가 없으면 {"missing": 사유}."""
    from tsinfer.config import db_path
    from tsinfer.data.preprocess import clean, to_wide

    tz = cfg.get("timezone", "Asia/Seoul")
    step = cfg["forecast"]["step_minutes"]
    path = Path(db) if db is not None else db_path()
    if not path.exists():
        return {"missing": f"수집기 DB 없음 ({path.name})"}
    since_utc = (pd.Timestamp(since, tz=tz).tz_convert("UTC") if since else pd.Timestamp("1970-01-01", tz="UTC"))
    try:
        pir = _load_db(path, "SELECT ts, device, value FROM readings WHERE space = ? AND metric = ? AND ts >= ?",
                       (space, PIR_METRIC, since_utc.isoformat()))
    except Exception as e:
        return {"missing": f"DB 읽기 실패 ({type(e).__name__}: {e})"}
    if pir.empty:
        return {"missing": f"{space} {PIR_METRIC} 기록 없음 ({since or '전체'} 이후)"}
    bins = pir_bins(pir, step)
    t0, t1 = bins.index.min(), bins.index.max()

    # 센서 추정 입력: 같은 기간 수집기 DB 값 (3시간 최솟값 특성 때문에 4시간 앞부터). 없으면 InfluxDB 내보내기 값
    sens = _load_db(path, "SELECT ts, space, device, metric, value FROM readings WHERE metric IN ({}) AND ts >= ? "
                    "AND ts <= ?".format(",".join("?" * len(cfg["metrics"]))),
                    (*cfg["metrics"], (t0 - pd.Timedelta(hours=4)).isoformat(),
                     (t1 + pd.Timedelta(minutes=step)).isoformat()))
    src, w = "수집기 DB", None
    if len(sens):
        sens["ts"] = pd.to_datetime(sens["ts"], utc=True, format="ISO8601")
        w = to_wide(clean(sens, cfg)[0], cfg)
    if w is None or (space, "co2") not in w or w[(space, "co2")].reindex(bins.index).notna().sum() == 0:
        src, w = "InfluxDB 내보내기", wide
    outdoor = cfg.get("model", {}).get("outdoor_space")
    f = st.features(w, space, outdoor, tz, step, cfg.get("holidays", [])).reindex(bins.index)
    sched_occ = st.occupancy_labels(bins.index, [space], sched, observations, tz)[space]
    ok_co2 = f["co2"].notna().to_numpy()
    est = {"규칙": np.where(ok_co2, st.rule_occupied(f), np.nan)}
    model, fx, label = occupancy_estimator(df, wide, cfg, state_model_path)
    if model is not None:
        est[label] = np.where(ok_co2, model.predict_proba(f[fx])[:, 1] >= 0.5, np.nan)
    frame = pd.DataFrame({"pir": bins, "sched": sched_occ, **{k: v for k, v in est.items()}}, index=bins.index)
    frame = frame[frame["pir"].notna() & frame["sched"].notna()]

    rows = []
    for name in ["시간표"] + list(est):
        col = "sched" if name == "시간표" else name
        sub = frame[frame[col].notna()]
        if sub.empty:
            rows.append({"estimator": name, "n": 0})
            continue
        y, p = sub["pir"].to_numpy().astype(int), sub[col].to_numpy().astype(int)
        s = st.scores(y, p)
        rows.append({"estimator": name, "n": s["n"], "agree_%": round(100 * float((y == p).mean()), 1),
                     "f1": round(2 * s["tp"] / max(2 * s["tp"] + s["fp"] + s["fn"], 1), 3),
                     "macro_f1": s["macro_f1"], "recall": s["recall"], "precision": s["precision"],
                     "empty_as_occupied_%": s["empty_as_occupied_%"],
                     "agree_sched_%": round(100 * float((sub["sched"].to_numpy().astype(int) == p).mean()), 1)})
    local_h = frame.index.tz_convert(tz).hour
    kinds = {"시간표=사용인데 PIR 없음": (frame["sched"] == 1) & (frame["pir"] == 0),
             "시간표=빔인데 PIR 있음": (frame["sched"] == 0) & (frame["pir"] == 1)}
    counts = pd.DataFrame([{"kind": k, "bins": int(m.sum()), "minutes": int(m.sum()) * step,
                            "share_%": round(100 * float(m.mean()), 1) if len(m) else 0.0} for k, m in kinds.items()])
    by_hour = pd.DataFrame({k: pd.Series(m.to_numpy(), index=local_h).groupby(level=0).sum()
                            for k, m in kinds.items()}).T.fillna(0).astype(int)
    by_hour = by_hour.loc[:, by_hour.sum() > 0]               # 어긋난 칸이 있는 시각만
    by_hour.columns.name = "시(한국)"
    return {"table": pd.DataFrame(rows).set_index("estimator"), "counts": counts, "by_hour": by_hour,
            "info": {"from": t0.tz_convert(tz), "to": (t1 + pd.Timedelta(minutes=step)).tz_convert(tz),
                     "bins": int(len(frame)), "pir_rows": int(len(pir)), "pir_active_%": round(100 * float(frame["pir"].mean()), 1)
                     if len(frame) else 0.0, "devices": sorted(pir["device"].astype(str).unique()), "sensor_src": src,
                     "space": space}}


def pir_lines(res: dict[str, Any]) -> list[str]:
    lines = ["## 5. 직접 관찰 대용 — PIR 활동 vs 시간표 vs 센서 추정", ""]
    if "missing" in res:
        return lines + [f"데이터 없음 ({res['missing']})", ""]
    i = res["info"]
    lines += [f"- 기간: {i['from']:%m-%d %H:%M} ~ {i['to']:%m-%d %H:%M} (한국 시간), {i['space']}, 5분 칸 {i['bins']}개 "
              f"(PIR 기록 {i['pir_rows']}건, 장치 {', '.join(i['devices'])}, 활동 칸 {i['pir_active_%']}%)",
              f"- PIR 활동 = 그 칸에 pir_activity > 0 인 보고가 있음 (기록은 직전 보고 이후 구간으로 펼침). "
              f"센서 추정 입력: {i['sensor_src']}",
              "- 아래 표는 PIR 을 정답(양성 = 활동 있음)으로 놓은 값. agree_% = 일치율(정확도), "
              "empty_as_occupied_% = PIR 없음인데 '사용 중' 비율, agree_sched_% = 시간표와의 일치율",
              "", res["table"].to_markdown(), "",
              "### 시간표와 PIR 이 어긋난 칸", "", res["counts"].to_markdown(index=False), "",
              "### 시각(한국 시간)별 어긋난 칸 수", "", res["by_hour"].to_markdown(), "",
              "- 한계: PIR 은 움직임만 잡아 가만히 앉아 있으면 '없음', 센서 시야 밖 활동도 '없음'이라 진짜 재실과 다를 수 있다 "
              "(정답이 아니라 대용 지표).", "",
              "### 해석 (직접 작성)", "- 시간표·센서 추정 중 무엇이 PIR 과 더 맞나, 어긋난 시간대의 실제 모습:", ""]
    return lines


# ---------------------------------------------------------------- 실행

def extra_sections(cfg: dict[str, Any], wide: pd.DataFrame, sched: dict[str, Any], df: pd.DataFrame,
                   observations=None, gru_path: Path | None = None, db: Path | None = None,
                   state_model_path: Path | None = None, folds: int = st.CV_FOLDS,
                   pir_since: str | None = PIR_SINCE, pir_space: str = TIMETABLE_SPACE) -> list[str]:
    """4·5절 보고서 줄. 한 절이 실패해도 다른 절과 1~3절 보고서는 그대로 나오게 한다."""
    import time
    import traceback

    lines: list[str] = []
    for title, fn in (
        ("## 4. 환기 경보 — 예측 모델로 판단 vs 분류기 따로",
         lambda: forecast_alarm_lines(forecast_alarm(cfg, wide, df, gru_path, folds), cfg)),
        ("## 5. 직접 관찰 대용 — PIR 활동 vs 시간표 vs 센서 추정",
         lambda: pir_lines(pir_compare(cfg, sched, wide, df, observations, db, pir_space, pir_since,
                                       state_model_path=state_model_path))),
    ):
        t0 = time.perf_counter()
        try:
            part = fn()
        except Exception as e:
            traceback.print_exc()
            part = [title, "", f"(실행 오류로 건너뜀: {type(e).__name__}: {e})", ""]
        part[2:2] = [f"- 이 절 실행 시간 {time.perf_counter() - t0:.0f}s", ""]
        lines += part
    return lines


def run(cfg: dict[str, Any], wide: pd.DataFrame, sched: dict[str, Any], observations=None,
        space: str = TIMETABLE_SPACE, **loso_kw):
    df = st.dataset(wide, cfg, sched, observations)
    tables, held = loso(cfg, wide, sched, observations, df=df, **loso_kw)
    events = mismatch_report(df, wide, cfg, space, held.get(space)) if space in set(df["space"]) else {}
    return tables, events, df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=TIMETABLE_SPACE, help="시간표 대조 규칙을 적용할 공간 (시간표가 확정된 곳)")
    ap.add_argument("--gru", type=Path, default=None, help="환기 경보 예측 기반에 쓸 GRU (기본 models/forecast_gru.pt)")
    ap.add_argument("--db", type=Path, default=None, help="PIR 을 읽을 수집기 DB (기본 .env 의 DB_PATH)")
    args = ap.parse_args()

    import time

    from tsinfer.data.preprocess import clean, load_raw, to_wide
    from tsinfer.data.split import split_by_time

    t0 = time.perf_counter()
    cfg = load_yaml("default.yaml")
    sched = load_yaml("schedule.yaml")
    obs_path = ROOT / (sched.get("observations_csv") or "data/observations.csv")
    observations = pd.read_csv(obs_path) if obs_path.exists() else None
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    tables, events, df = run(cfg, wide, sched, observations, args.space)
    t_base = time.perf_counter() - t0
    extra = extra_sections(cfg, wide, sched, df, observations, gru_path=args.gru, db=args.db)
    _, _, test = split_by_time(wide, cfg)
    days = (test.index.max() - test.index.min()) / pd.Timedelta("1D")
    tz = cfg.get("timezone", "Asia/Seoul")
    cols = ["macro_f1", "empty_as_occupied_%", "recall", "precision", "positive_%", "n", "fp", "fn", "threshold"]

    lines = [
        "# 과제 5 심화. 학습하지 않은 공간 · 시간표 대조",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터, 학습 ~{cfg['split']['train_end']}, 검증 ~{cfg['split']['valid_end']}, "
        f"시험 {test.index.min():%Y-%m-%d} ~ {test.index.max():%Y-%m-%d} (UTC), 5분 간격. 공간: {', '.join(df['space'].unique())}",
        f"- 사용 여부 정답: configs/schedule.yaml (source={sched.get('source')}), 직접 관찰 "
        f"{0 if observations is None else len(observations)}건. 입력·임계값·분류기는 과제 5(state.py)와 같음",
        "- 공간 하나 빼고 학습: 나머지 공간의 학습 기간(환기 경보는 학습+검증)으로 학습 → 빠진 공간의 시험 기간 채점. "
        "'같은 공간 포함' 행은 모든 공간으로 학습한 같은 모델을 그 공간 시험 행만 채점한 것 (과제 5 표를 공간별로 나눈 값)",
        "- 환기 경보 임계값: 학습 공간들의 학습+검증 기간을 시간순 5구간 교차 예측해 재현율 ≥85% 인 가장 높은 확률값",
        f"- 시간표 대조 규칙: {args.space}, 시간표를 {SLOT_MIN}분 칸으로 잘라 칸 시작 후 {NOSHOW_MIN}분 내내 '비어 있음'이면 "
        f"'수업인데 비어 있음', 시간표 밖에서 '사용 중'이 {UNSCHEDULED_MIN}분 이상 이어지면 '수업 아닌데 사용'",
        f"- 실행 시간 {time.perf_counter() - t0:.0f}s (1~3절 {t_base:.0f}s, 4·5절 {time.perf_counter() - t0 - t_base:.0f}s)",
        "",
    ]
    if sched.get("note"):
        lines += [f"> 정답 출처: {sched['note']}", ""]
    for task, title in (("occupied", "1. 사용 여부 — 공간 하나 빼고 학습"), ("ventilation", "2. 환기 경보 — 공간 하나 빼고 학습")):
        if task not in tables:
            lines += [f"## {title}", "", "(정답이 한 종류뿐이거나 시험 행이 없어 건너뜀)", ""]
            continue
        t = tables[task]
        lines += [f"## {title}", "", "### 공간 평균", "", loso_summary(t).to_markdown(), "",
                  "### 공간별", "",
                  t.set_index(["space", "method"]).reindex(columns=[c for c in cols if c in t.columns]).to_markdown(), ""]
    lines += [f"## 3. 시간표 대조 규칙 ({args.space}, 시험 기간 {days:.1f}일)", ""]
    if events:
        lines += ["### 건수", "", mismatch_counts(events, days).to_markdown(index=False), "",
                  "### 시작 시각(한국 시간)별 건수", "", mismatch_by_hour(events).to_markdown(), ""]
        for name, ev in events.items():
            if len(ev):
                show = ev.sort_values("start").head(15).copy()
                show["start"] = pd.to_datetime(show["start"]).map(lambda t: t.tz_convert(tz).strftime("%m-%d(%a) %H:%M"))
                show["end"] = pd.to_datetime(show["end"]).map(lambda t: t.tz_convert(tz).strftime("%H:%M"))
                lines += [f"### 사건 목록 (앞 15건) — {name}", "", show.to_markdown(index=False), ""]
    else:
        lines += [f"{args.space} 데이터가 없어 건너뜀", ""]
    lines += [
        "### 한계",
        "- 회의실 노쇼(예약 후 15분간 비어 있음)·예약 없는 사용은 예약표·직접 관찰 기록이 없어 만들지 못했다. "
        "위 규칙은 같은 판정 틀을 시간표가 확정된 실습실에 적용한 대체물이다.",
        "- 직접 관찰 기록과의 대조는 하지 못했다 (관찰 0건). 사건 목록을 실제 기억·출입 기록과 맞춰 보는 것이 다음 단계다.",
        "- 순환 논리: 사용 여부 정답 자체가 시간표라, '수업인데 비어 있음'은 곧 추정기의 놓침(FN), "
        "'수업 아닌데 사용'은 헛경보(FP)와 같은 것을 다른 이름으로 센 것이다. 정답이 시간표인 한 '시간표가 틀렸다'와 "
        "'추정기가 틀렸다'를 구분할 수 없다. hgb 추정기는 이 공간의 정답을 학습에 쓰지 않아 순환을 조금 줄인 것일 뿐이다.",
        "- 공휴일(holiday=True)에 나온 '수업인데 비어 있음'은 시간표에 휴일 예외가 없어서 생긴, 실제로 맞는 탐지일 가능성이 크다.",
        "",
        *extra,
        "## 해석 (직접 작성)",
        "- 학습하지 않은 공간에 바로 쓸 수 있는 수준인가 (macro-F1 이 얼마나 떨어졌고, 어느 공간에서 크게 떨어졌나):",
        "- 떨어진 이유 (방 크기·환기·인원 차이로 CO₂ 수준이 달라서 등)와 줄이는 방법:",
        "- 시간표 대조 사건 중 실제와 맞는 것 / 아닌 것 (점심시간·휴일 등):",
        "",
    ]
    out = ROOT / "reports" / "05b_state_deep.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
