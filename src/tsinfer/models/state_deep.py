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
"""
from __future__ import annotations

import argparse
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


# ---------------------------------------------------------------- 실행

def run(cfg: dict[str, Any], wide: pd.DataFrame, sched: dict[str, Any], observations=None,
        space: str = TIMETABLE_SPACE, **loso_kw):
    df = st.dataset(wide, cfg, sched, observations)
    tables, held = loso(cfg, wide, sched, observations, df=df, **loso_kw)
    events = mismatch_report(df, wide, cfg, space, held.get(space)) if space in set(df["space"]) else {}
    return tables, events, df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default=TIMETABLE_SPACE, help="시간표 대조 규칙을 적용할 공간 (시간표가 확정된 곳)")
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
        f"- 실행 시간 {time.perf_counter() - t0:.0f}s",
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
