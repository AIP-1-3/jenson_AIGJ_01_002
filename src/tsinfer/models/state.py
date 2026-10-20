"""과제 5: 상태 추정 — 사용 여부, 환기 경보.

    python -m tsinfer.models.state        # → models/state_*.joblib, reports/05_state.md

정답
- 사용 여부: configs/schedule.yaml (시간표 → 예외 → 직접 관찰 순으로 덮어씀). 입력에는 넣지 않는다.
- 환기 경보: 지금 CO₂ < 1,000 이고 앞으로 30분 안에 1,000 이상이 되면 양성.
  이미 1,000 이상인 시점은 '미리 알려 주는' 문제가 아니므로 평가에서 뺀다.

입력 (센서만): 최근 1시간 CO₂·온도·습도 수준, 15·30·60분 변화량, 최근 1시간 CO₂ 최댓값·표준편차,
외기 온도, 한국 시간 시각(sin/cos)·평일 여부.

분류기 2종: 로지스틱 회귀(선형, 해석 쉬움) / 히스토그램 그래디언트 부스팅(비선형).
둘 다 불균형을 class_weight=balanced 로 보정한다.
환기 경보 임계값은 검증 기간 재현율 ≥85% 를 만족하는 가장 높은 확률값으로 정한다 (헛경보 최소화).
"""
from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml

DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
RECALL_TARGET = 0.85


# ---------------------------------------------------------------- 정답

def _hm(s: str) -> float:
    h, m = s.split(":")
    return int(h) + int(m) / 60


def occupancy_labels(index: pd.DatetimeIndex, spaces: list[str], sched: dict[str, Any],
                     observations: pd.DataFrame | None = None, tz: str = "Asia/Seoul") -> pd.DataFrame:
    """시점 × 공간 0/1. 시간표에 없는 공간은 NaN (채점 제외)."""
    local = index.tz_convert(tz)
    hour = local.hour + local.minute / 60
    out = pd.DataFrame(np.nan, index=index, columns=spaces)
    for sp in spaces:
        rules = (sched.get("weekly") or {}).get(sp)
        if rules is None:
            continue
        occ = np.zeros(len(index), bool)
        for r in rules:
            days = [DAYS[d] for d in r["days"]]
            occ |= np.isin(local.dayofweek, days) & (hour >= _hm(r["start"])) & (hour < _hm(r["end"]))
        out[sp] = occ.astype(float)
    for ex in sched.get("exceptions") or []:
        if ex["space"] not in out:
            continue
        d = pd.Timestamp(ex["date"]).date()
        sel = (local.date == d) & (hour >= _hm(ex["start"])) & (hour < _hm(ex["end"]))
        out.loc[sel, ex["space"]] = float(ex.get("occupied", False))
    if observations is not None and len(observations):
        for r in observations.itertuples():
            if r.space not in out:
                continue
            s = pd.Timestamp(r.start, tz=tz) if pd.Timestamp(r.start).tzinfo is None else pd.Timestamp(r.start)
            e = pd.Timestamp(r.end, tz=tz) if pd.Timestamp(r.end).tzinfo is None else pd.Timestamp(r.end)
            out.loc[(index >= s) & (index < e), r.space] = float(bool(r.occupied))
    return out


def ventilation_labels(co2: pd.Series, threshold: float, within_steps: int) -> pd.Series:
    """1: 지금 < 임계값 이고 다음 within_steps 칸 안에 ≥ 임계값. 이미 넘은 시점·미래가 비면 NaN."""
    fut = pd.concat([co2.shift(-k) for k in range(1, within_steps + 1)], axis=1)
    fut_max = fut.max(axis=1, skipna=False)
    y = (fut_max >= threshold).astype(float)
    y[(co2 >= threshold) | co2.isna() | fut_max.isna()] = np.nan
    return y


# ---------------------------------------------------------------- 입력 특성 (센서만)

def features(wide: pd.DataFrame, space: str, outdoor: str | None, tz: str, step_min: int = 5,
             holidays: list[str] | tuple = ()) -> pd.DataFrame:
    per_h = 60 // step_min
    f = pd.DataFrame(index=wide.index)
    for m in ("co2", "temperature", "humidity"):
        s = wide[(space, m)]
        f[m] = s
        for k in (15, 30, 60):
            f[f"{m}_d{k}"] = s - s.shift(k // step_min)
    co2 = wide[(space, "co2")]
    f["co2_max1h"] = co2.rolling(per_h, min_periods=per_h // 2).max()
    f["co2_std1h"] = co2.rolling(per_h, min_periods=per_h // 2).std()
    f["co2_min3h"] = co2.rolling(3 * per_h, min_periods=per_h).min()      # 그 방의 '빈방 수준' 추정
    f["co2_above_min"] = co2 - f["co2_min3h"]
    f["out_temp"] = wide[(outdoor, "temperature")] if outdoor and (outdoor, "temperature") in wide else np.nan
    local = wide.index.tz_convert(tz)
    hour = local.hour + local.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    hol = np.isin(local.strftime("%Y-%m-%d"), [str(d) for d in holidays])
    f["weekday"] = ((local.dayofweek < 5) & ~hol).astype(float)
    return f


def dataset(wide: pd.DataFrame, cfg: dict[str, Any], sched: dict[str, Any], observations=None) -> pd.DataFrame:
    """긴 형식: time, space, 특성…, y_occ, y_vent."""
    tz = cfg.get("timezone", "Asia/Seoul")
    step = cfg["forecast"]["step_minutes"]
    th = cfg["thresholds"]
    spaces = [sp for sp in cfg["spaces"] if (sp, "co2") in wide and wide[(sp, "co2")].notna().any()]
    occ = occupancy_labels(wide.index, spaces, sched, observations, tz)
    outdoor = cfg.get("model", {}).get("outdoor_space")
    parts = []
    for sp in spaces:
        f = features(wide, sp, outdoor, tz, step, cfg.get("holidays", []))
        f["space"] = sp
        f["y_occ"] = occ[sp]
        f["y_vent"] = ventilation_labels(wide[(sp, "co2")], th["co2_ventilate_ppm"],
                                         th["ventilation_alert_within_minutes"] // step)
        parts.append(f.rename_axis("time").reset_index())
    return pd.concat(parts, ignore_index=True)


FEATURES_EXCLUDE = {"time", "space", "y_occ", "y_vent"}


# ---------------------------------------------------------------- 규칙 기준선 (서비스와 같은 규칙)

def rule_occupied(df: pd.DataFrame) -> np.ndarray:
    """서비스 v0 규칙: 15분 CO₂ 상승 ≥30ppm. 오래 머문 뒤 평형이면 놓친다 → 빈방 수준 대비 +150 도 사용중으로."""
    return ((df["co2_d15"] >= 30) | (df["co2_above_min"] >= 150)).to_numpy()


def rule_vent(df: pd.DataFrame, threshold: float, within_min: int) -> np.ndarray:
    """최근 30분 기울기를 그대로 늘려 30분 안에 임계값을 넘으면 경보."""
    slope = df["co2_d30"] / 30.0
    return (df["co2"] + slope * within_min >= threshold).to_numpy()


# ---------------------------------------------------------------- 학습·평가

def make_models(seed: int = 0) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logreg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(class_weight="balanced", max_iter=2000)),
        "hgb": HistGradientBoostingClassifier(class_weight="balanced", max_iter=300, learning_rate=0.05,
                                              early_stopping=True, random_state=seed),
    }


def pick_threshold(y: np.ndarray, p: np.ndarray, recall_target: float = RECALL_TARGET) -> float:
    """재현율 ≥ 목표 를 만족하는 가장 높은 확률 임계값 (헛경보를 가장 줄이는 값)."""
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - recall_target) * len(pos)))
    return float(pos[k]) if k < len(pos) else 0.5


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)), "positive_%": round(100 * float(np.mean(y)), 1),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "macro_f1": round(f1_score(y, pred, average="macro", zero_division=0), 3),
        "empty_as_occupied_%": round(100 * fp / max(tn + fp, 1), 1),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def error_cases(df: pd.DataFrame, y: np.ndarray, pred: np.ndarray, tz: str, top: int = 8) -> pd.DataFrame:
    """틀린 시점을 (공간, 시각대, 틀린 방향)으로 묶어 많은 순."""
    wrong = y != pred
    if not wrong.any():
        return pd.DataFrame()
    d = df.loc[wrong, ["space", "time", "co2", "co2_d15"]].copy()
    d["hour"] = pd.to_datetime(d["time"]).dt.tz_convert(tz).dt.hour
    d["kind"] = np.where(pred[wrong] == 1, "헛경보(FP)", "놓침(FN)")
    g = d.groupby(["space", "hour", "kind"]).agg(n=("co2", "size"), co2_mean=("co2", "mean"),
                                                  co2_d15_mean=("co2_d15", "mean"))
    return g.sort_values("n", ascending=False).head(top).round(1).reset_index()


def run(cfg: dict[str, Any], wide: pd.DataFrame, sched: dict[str, Any], observations=None):
    from tsinfer.data.split import split_by_time

    tz = cfg.get("timezone", "Asia/Seoul")
    th = cfg["thresholds"]
    df = dataset(wide, cfg, sched, observations)
    tr_idx, va_idx, te_idx = (s.index for s in split_by_time(wide, cfg))
    part = {"train": df["time"].isin(tr_idx), "valid": df["time"].isin(va_idx), "test": df["time"].isin(te_idx)}
    feats = [c for c in df.columns if c not in FEATURES_EXCLUDE]
    results, fitted, errors, thresholds = {}, {}, {}, {}

    for task, ycol in (("occupied", "y_occ"), ("ventilation", "y_vent")):
        ok = df[ycol].notna() & df["co2"].notna()
        tr, va, te = (df[ok & part[k]] for k in ("train", "valid", "test"))
        if tr[ycol].nunique() < 2 or te.empty:
            continue
        rows = {}
        rule = rule_occupied(te) if task == "occupied" else rule_vent(te, th["co2_ventilate_ppm"],
                                                                       th["ventilation_alert_within_minutes"])
        rows["rule"] = scores(te[ycol].to_numpy().astype(int), rule.astype(int))
        for name, model in make_models(cfg.get("model", {}).get("seed", 0)).items():
            model.fit(tr[feats], tr[ycol].astype(int))
            thr = 0.5
            if task == "ventilation" and len(va) and va[ycol].sum() > 0:
                thr = pick_threshold(va[ycol].to_numpy().astype(int), model.predict_proba(va[feats])[:, 1])
            p = model.predict_proba(te[feats])[:, 1]
            pred = (p >= thr).astype(int)
            rows[name] = {**scores(te[ycol].to_numpy().astype(int), pred), "threshold": round(thr, 3)}
            fitted[f"{task}_{name}"] = model
            thresholds[f"{task}_{name}"] = thr
            errors[f"{task}_{name}"] = error_cases(te, te[ycol].to_numpy().astype(int), pred, tz)
        results[task] = pd.DataFrame(rows).T
    return results, fitted, thresholds, errors, feats, df


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    import joblib

    from tsinfer.data.preprocess import clean, load_raw, to_wide
    from tsinfer.data.split import split_by_time

    cfg = load_yaml("default.yaml")
    sched = load_yaml("schedule.yaml")
    obs_path = ROOT / (sched.get("observations_csv") or "data/observations.csv")
    observations = pd.read_csv(obs_path) if obs_path.exists() else None
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    results, fitted, thresholds, errors, feats, df = run(cfg, wide, sched, observations)

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"models": fitted, "thresholds": thresholds, "features": feats}, ROOT / "models" / "state.joblib")

    _, _, test = split_by_time(wide, cfg)
    cols = ["n", "positive_%", "recall", "precision", "macro_f1", "empty_as_occupied_%", "tn", "fp", "fn", "tp", "threshold"]
    lines = [
        "# 과제 5. 상태 추정",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터, 학습 ~{cfg['split']['train_end']}, 검증 ~{cfg['split']['valid_end']}, "
        f"시험 {test.index.min():%Y-%m-%d} ~ {test.index.max():%Y-%m-%d} (UTC), 5분 간격",
        f"- 사용 여부 정답: configs/schedule.yaml (source={sched.get('source')}), "
        f"직접 관찰 {0 if observations is None else len(observations)}건 우선 적용. 정답 정보는 입력에 넣지 않음",
        f"- 환기 경보 정답: 지금 <{cfg['thresholds']['co2_ventilate_ppm']}ppm 이고 "
        f"{cfg['thresholds']['ventilation_alert_within_minutes']}분 안에 ≥{cfg['thresholds']['co2_ventilate_ppm']}ppm. 이미 넘은 시점 제외",
        f"- 입력 특성 {len(feats)}개: {', '.join(feats)}",
        f"- 환기 경보 임계값: 검증 기간 재현율 ≥{RECALL_TARGET:.0%} 인 가장 높은 확률값 / 사용 여부는 0.5",
        "",
    ]
    if sched.get("source") == "placeholder":
        lines += ["> ⚠ 사용 여부 정답이 임시 가정(평일 9~18시)이다. 실제 시간표로 바꾸면 다시 실행한다.", ""]
    for task, title in (("occupied", "사용 여부"), ("ventilation", "환기 경보 (30분 전)")):
        if task not in results:
            lines += [f"## {title}", "", "(정답이 한 종류뿐이라 학습 불가 — 정답 설정 확인)", ""]
            continue
        r = results[task].reindex(columns=[c for c in cols if c in results[task].columns])
        lines += [f"## {title}", "", r.to_markdown(), ""]
        for name in ("logreg", "hgb"):
            e = errors.get(f"{task}_{name}")
            if e is not None and len(e):
                lines += [f"### 가장 많이 틀린 상황 — {name}", "", e.to_markdown(index=False), ""]
    lines += [
        "## 확인 기준",
        f"- 환기 경보 재현율 ≥{RECALL_TARGET:.0%}, 규칙보다 macro-F1 이 높을 것, 빈방→사용중 오류율(empty_as_occupied_%) 따로 보고",
        "",
        "## 해석 (직접 작성)",
        "- 두 분류기 차이와 이유:",
        "- 가장 많이 틀린 상황의 실제 모습 (문 열린 회의, 짧은 방문, 사람이 나간 직후 등):",
        "- 서비스에서 어떤 임계값으로 쓸지:",
        "",
    ]
    out = ROOT / "reports" / "05_state.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
