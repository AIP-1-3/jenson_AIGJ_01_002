"""과제 4 심화: 예측 오차 탐지기, 영점 틀어짐(drift) 전용 탐지, 결합 조합 선택.

    python -m tsinfer.models.anomaly_deep    # → reports/04b_anomaly_deep.md (models/forecast_gru.pt, models/anomaly_ae.pt 재사용)

과제 4(anomaly.py)와 같은 고장 주입(seed 0, 종류별 15건)·같은 채점으로 새 탐지기를 비교한다.
새 탐지기는 모두 시험 시점에 과거만 본다 (rolling 은 뒤쪽 창, 예측은 이미 지난 시점에서 낸 것).
  pe     : 예측 오차 — GRU 가 5·15분 전에 낸 예측과 지금 값의 차이 (시각별 평균 절대 오차로 나눔). CO₂·온도만
  cusum  : 예측 오차 누적 — 부호 있는 오차를 CUSUM 으로 쌓는다. 천천히 어긋나는 고장은 한 칸 오차가 작아도 쌓인다
  floor  : 바닥 CO₂ — 최근 2시간 최솟값이 '그 시각·평일/휴일에 정상이면 넘지 않던 값' 위로 올라감 (빈방 CO₂ ≈ 외기)
  cross  : 공간 간 비교 — 온도·습도에서 (내 값 - 다른 공간 중앙값)이 6시간 사이 얼마나 변했나
  stuck2 : 교차 값 고정 — 온도·습도가 1시간 넘게 같은 값인데 같은 공간 다른 항목은 크게 움직임
임계값: 탐지기마다 검증 기간(고장 없음) 경보 사건이 하루 0.5 / 0.25 / 0.1 회 이하가 되는 가장 낮은 값.
결합 선택: 검증 기간에 따로 넣은 고장(seed 1)으로 '규칙 OR 오토인코더 + 새 탐지기 조합'을 모두 채점해
검증 헛경보 ≤1회/일 안에서 탐지율이 가장 높은 조합을 고른다. 시험 기간은 고르는 데 쓰지 않는다.

과제 안내 고장 5종 시험: 과제 안내의 고장은 튄 값·값 고정·영점 틀어짐·냉방 정지·환기 정지다.
anomaly.py 는 냉방 정지·환기 정지 대신 noise·dropout 을 넣으므로, 시험 기간에 따로 다음을 만들어 채점한다.
  spike/stuck/drift : 과제 4 시험 주입(seed 0)의 같은 고장을 그대로 (noise·dropout 자리는 원본으로 되돌림)
  cooling_off : 평일 9~18시(한국 시각, 공휴일 제외) 사용 시간 안, 2~3시간에 걸쳐 온도 +3~5℃ 선형 (끝에서 원래로)
  vent_off    : 평일 사용 시간 중 CO₂ 가 오르는 구간, 시작 시점 대비 상승분을 1.8배로 (2~3시간)
냉방·환기 고장은 실내 공간에만 넣는다 (외기 공간 제외). 임계값·조합은 검증 기간에서 정한 것을 그대로 쓴다.
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import anomaly as an
from tsinfer.models.anomaly import GRACE, KINDS, METRICS, Fault

HORIZONS = (1, 3)        # 예측 오차 지평 (칸): 5·15분
Z_CLIP = 5.0             # CUSUM 에 넣는 한 칸 오차 상한 — 튐 하나가 누적을 오래 붙잡지 않게
CUSUM_K = 0.25           # CUSUM 허용치 (오차 단위). 드리프트 CO₂ 는 한 칸 +3.5ppm 이라 작게 둔다
FLOOR_WIN = 24           # 바닥 CO₂: 최근 2시간 최솟값
FLOOR_Q = 0.95           # 학습 기간 (평일/휴일 × 한국 시각)별 이 분위수를 '정상 바닥 상한'으로
CROSS_SMOOTH = 12        # 공간 간 차이를 1시간 중앙값으로 다듬은 뒤
CROSS_LAG = 72           # 6시간 전과 비교 (드리프트 길이)
STUCK2_MIN = 12          # 교차 값 고정: 1시간 이상 같은 값일 때부터 본다
BUDGETS = (0.5, 0.25, 0.1)   # 탐지기 하나의 검증 기간 경보 예산 (회/일)
FA_LIMIT = 1.0           # 결합의 검증 기간 헛경보 상한 (회/일)
VALID_SEED = 1           # 검증 기간 고장 주입 seed (시험 seed 0 과 다르게)
QGRID = (0.9, 0.95, 0.97, 0.98, 0.99, 0.993, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 1.0)
TARGETS = ("co2", "temperature")     # GRU 예측 항목
PAIR_METRICS = ("temperature", "humidity")
LABELS = {"rules": "규칙", "ae": "오토인코더", "pe": "예측오차", "cusum": "예측오차 누적",
          "floor": "바닥 CO₂", "cross": "공간 간 비교", "stuck2": "교차 값 고정"}
APPLIES = {"pe": TARGETS, "cusum": TARGETS, "floor": ("co2",), "cross": PAIR_METRICS, "stuck2": PAIR_METRICS}

Scores = dict[str, pd.DataFrame]     # 공간 → (시점 × 항목) 이상 점수, 해당 없는 항목은 NaN

# 과제 안내 고장 5종 시험
T4_KINDS = ("spike", "stuck", "drift", "cooling_off", "vent_off")
T4_LABELS = {"spike": "튄 값", "stuck": "값 고정", "drift": "영점 틀어짐",
             "cooling_off": "냉방 정지", "vent_off": "환기 정지"}
USE_HOURS = (9, 18)      # 사용 시간 (한국 시각, 평일) — configs/schedule.yaml 의 평일 9~18시와 같음
USAGE_HOURS = (2, 3)     # 냉방·환기 정지 지속 (시간, 양 끝 포함, 5분 단위 무작위)
COOL_AMP = (3.0, 5.0)    # 냉방 정지: 끝 시점 온도 상승 (℃)
VENT_GAIN = 1.8          # 환기 정지: 시작 시점 대비 CO₂ 상승분 배율
VENT_MIN_RISE = 100.0    # 환기 정지 자리: 원본 CO₂ 가 구간 끝에서 시작보다 이만큼(ppm) 이상 올라 있어야 '오르는 구간'
T4_SEP = pd.Timedelta(hours=2)   # 새 고장과 다른 고장 사이 간격. 사용 시간이 하루 9시간뿐이라 과제 4 의 6시간보다 짧게
DETECTOR_LABELS = {"spike": "규칙-급변", "stuck": "규칙-값 고정", "range": "규칙-범위 밖", "noise": "규칙-잡음",
                   "gap": "규칙-끊김", "ae": "오토인코더", **LABELS}


# ---------------------------------------------------------------- 공통

def local_hours(index: pd.DatetimeIndex, tz: str) -> np.ndarray:
    return np.asarray(index.tz_convert(tz).hour)


def workdays(index: pd.DatetimeIndex, tz: str, holidays: list[str]) -> np.ndarray:
    local = index.tz_convert(tz)
    hol = np.isin(local.strftime("%Y-%m-%d"), list(holidays)) if holidays else np.zeros(len(index), bool)
    return np.asarray((local.dayofweek < 5) & ~hol)


def empty_scores(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=index, columns=list(METRICS))


def score_flags(scores: Scores, thr: float, name: str) -> dict[str, pd.DataFrame]:
    """점수 > 임계값 → anomaly.evaluate 가 읽는 '<이름>:<항목>' 불리언 열."""
    return {sp: (df > thr).fillna(False).astype(bool).rename(columns=lambda m: f"{name}:{m}")
            for sp, df in scores.items()}


def pick_score_threshold(scores: Scores, days: float, budget: float = 0.5) -> float:
    """정상(검증) 기간 경보 사건이 하루 budget 이하가 되는 가장 낮은 임계값 (anomaly.pick_threshold 와 같은 규칙,
    분위수 칸만 더 촘촘히)."""
    vals = np.concatenate([df.to_numpy(dtype=float).ravel() for df in scores.values()]) if scores else np.zeros(0)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return np.inf
    for q in QGRID:
        thr = float(np.quantile(vals, q))
        n = sum(len(an.events((df > thr).any(axis=1))) for df in scores.values())
        if n / max(days, 1e-9) <= budget:
            return thr
    return float(vals.max())


# ---------------------------------------------------------------- 예측 오차 (GRU)

def gru_residuals(model, spec, wide: pd.DataFrame, horizons=HORIZONS) -> dict[tuple[str, str, int], pd.Series]:
    """(공간, 항목, 지평) → 실제값 - h칸 전에 낸 예측. 입력 창이 결측이라 예측이 없으면 NaN."""
    from tsinfer.models import forecast as fc

    x, _, info = fc.make_windows(wide, spec, with_target=False)
    out = {}
    pred = fc.decode(fc.predict_array(model, x), info, spec) if len(x) else None
    for sp in spec.spaces:
        for m in TARGETS:
            for h in horizons:
                p = pd.Series(np.nan, index=wide.index)
                if pred is not None:
                    sel = (info["space"] == sp).to_numpy()
                    t = pd.DatetimeIndex(info.loc[sel, "time"]) + pd.Timedelta(minutes=h * spec.step_minutes)
                    p = pd.Series(pred[m][sel, h - 1], index=t).reindex(wide.index)
                out[(sp, m, h)] = wide[(sp, m)] - p
    return out


def fit_residual_scale(res: dict[tuple[str, str, int], pd.Series], tz: str) -> dict[tuple[str, str, int], np.ndarray]:
    """한국 시각(0~23시)별 평균 절대 오차. 밤에는 오차가 작아 같은 어긋남도 더 크게 보이게 된다.
    너무 작은 칸은 전체 평균의 1/4 로 막는다 (분해능 때문에 0 이 되는 시각)."""
    out = {}
    for key, r in res.items():
        a = r.abs()
        g = float(a.mean())
        if not np.isfinite(g) or g <= 0:
            continue
        by = a.groupby(local_hours(a.index, tz)).mean().reindex(range(24)).fillna(g).to_numpy()
        out[key] = np.maximum(by, g / 4)
    return out


def norm_residuals(res, scale, sp: str, m: str, index: pd.DatetimeIndex, tz: str) -> list[pd.Series]:
    hr = local_hours(index, tz)
    return [res[(sp, m, h)] / scale[(sp, m, h)][hr] for h in HORIZONS if (sp, m, h) in res and (sp, m, h) in scale]


def pe_scores(res, scale, spaces: list[str], index: pd.DatetimeIndex, tz: str) -> Scores:
    """지평별 |정규화 오차| 의 최댓값. 습도는 GRU 가 예측하지 않아 NaN(해당 없음)."""
    out = {}
    for sp in spaces:
        f = empty_scores(index)
        for m in TARGETS:
            zs = norm_residuals(res, scale, sp, m, index, tz)
            if zs:
                f[m] = pd.concat(zs, axis=1).abs().max(axis=1, skipna=True)
        out[sp] = f
    return out


def cusum(z: np.ndarray, k: float = CUSUM_K, clip: float = Z_CLIP) -> np.ndarray:
    """양쪽 CUSUM. 결측 칸은 0 으로 보고 허용치만큼 줄어든다."""
    hi = lo = 0.0
    out = np.empty(len(z))
    for i, v in enumerate(np.clip(np.nan_to_num(z, nan=0.0), -clip, clip)):
        hi = max(0.0, hi + v - k)
        lo = max(0.0, lo - v - k)
        out[i] = max(hi, lo)
    return out


def cusum_scores(res, scale, spaces: list[str], index: pd.DatetimeIndex, tz: str) -> Scores:
    out = {}
    for sp in spaces:
        f = empty_scores(index)
        for m in TARGETS:
            zs = norm_residuals(res, scale, sp, m, index, tz)
            if zs:
                z = pd.concat(zs, axis=1).mean(axis=1, skipna=True).to_numpy()
                f[m] = cusum(z)
        out[sp] = f
    return out


# ---------------------------------------------------------------- 바닥 CO₂ (빈방이면 외기 근처)

def floor_series(wide: pd.DataFrame, sp: str) -> pd.Series:
    return wide[(sp, "co2")].rolling(FLOOR_WIN, min_periods=FLOOR_WIN // 2).min()


def fit_floor(wide_train: pd.DataFrame, spaces: list[str], tz: str, holidays: list[str]) -> dict[str, np.ndarray]:
    """공간별 (휴일0/평일1 × 한국 시각) 바닥 상한. 앞뒤 1시간과 최댓값을 취해 시각 경계에서 튀지 않게."""
    out = {}
    for sp in spaces:
        m = floor_series(wide_train, sp)
        allq = m.quantile(FLOOR_Q)
        if not np.isfinite(allq):
            continue
        q = m.groupby([workdays(m.index, tz, holidays).astype(int), local_hours(m.index, tz)]).quantile(FLOOR_Q)
        tab = np.full((2, 24), float(allq))
        for (w, h), v in q.items():
            if np.isfinite(v):
                tab[int(w), int(h)] = v
        out[sp] = np.maximum.reduce([np.roll(tab, 1, axis=1), tab, np.roll(tab, -1, axis=1)])
    return out


def floor_scores(wide: pd.DataFrame, spaces: list[str], table: dict[str, np.ndarray], tz: str,
                 holidays: list[str]) -> Scores:
    """점수 = 최근 2시간 최솟값 - 그 시각 정상 바닥 상한 (ppm). CO₂ 만."""
    wd = workdays(wide.index, tz, holidays).astype(int)
    hr = local_hours(wide.index, tz)
    out = {}
    for sp in spaces:
        f = empty_scores(wide.index)
        if sp in table:
            f["co2"] = floor_series(wide, sp).to_numpy() - table[sp][wd, hr]
        out[sp] = f
    return out


# ---------------------------------------------------------------- 공간 간 비교 (온도·습도)

def cross_change(wide: pd.DataFrame, spaces: list[str], sp: str, m: str) -> pd.Series | None:
    """(내 값 - 다른 공간 중앙값)의 1시간 중앙값이 6시간 전보다 얼마나 변했나. 건물 전체가 같이 변하면 0."""
    others = [o for o in spaces if o != sp and (o, m) in wide]
    if not others or (sp, m) not in wide:
        return None
    d = wide[(sp, m)] - wide[[(o, m) for o in others]].median(axis=1, skipna=True)
    d = d.rolling(CROSS_SMOOTH, min_periods=CROSS_SMOOTH // 2).median()   # 중앙값: 튐 하나가 6시간 뒤 '되울림'으로 다시 보이지 않게
    return d - d.shift(CROSS_LAG)


def fit_cross_scale(wide_train: pd.DataFrame, spaces: list[str]) -> dict[tuple[str, str], float]:
    out = {}
    for sp in spaces:
        for m in PAIR_METRICS:
            c = cross_change(wide_train, spaces, sp, m)
            if c is not None and np.isfinite(c.std()) and c.std() > 0:
                out[(sp, m)] = float(c.std())
    return out


def cross_scores(wide: pd.DataFrame, spaces: list[str], scale: dict[tuple[str, str], float]) -> Scores:
    """|변화| / 학습 기간 표준편차. 한 공간이 어긋나면 다른 공간의 차이도 따라 변하므로,
    같은 시점·항목에서 점수가 가장 큰 공간에만 점수를 남긴다 (어긋난 센서를 가리키게)."""
    out = {sp: empty_scores(wide.index) for sp in spaces}
    for m in PAIR_METRICS:
        cols = {sp: (cross_change(wide, spaces, sp, m).abs() / scale[(sp, m)])
                for sp in spaces if (sp, m) in scale}
        if not cols:
            continue
        df = pd.DataFrame(cols)
        top = df.max(axis=1, skipna=True)
        for sp in cols:
            out[sp][m] = df[sp].where(df[sp] >= top)
    return out


# ---------------------------------------------------------------- 교차 값 고정 (온도·습도)

def causal_run(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """지금까지 같은 값이 이어진 칸 수(미래를 보지 않음)와 구간 번호."""
    same = s.notna() & (s.diff() == 0)
    grp = (~same).cumsum()
    run = same.astype(int).groupby(grp).cumsum() + 1
    return run.where(s.notna(), 0), grp


def fit_move_scale(wide_train: pd.DataFrame, spaces: list[str]) -> dict[tuple[str, str], float]:
    """항목별 '1시간 동안 움직인 폭'의 학습 기간 중앙값 — 다른 항목이 얼마나 움직였는지 재는 단위."""
    out = {}
    for sp in spaces:
        for m in METRICS:
            if (sp, m) not in wide_train:
                continue
            s = wide_train[(sp, m)]
            r = (s.rolling(12, min_periods=6).max() - s.rolling(12, min_periods=6).min()).median()
            out[(sp, m)] = float(r) if np.isfinite(r) and r > 0 else 1.0
    return out


def stuck2_scores(wide: pd.DataFrame, spaces: list[str], move: dict[tuple[str, str], float]) -> Scores:
    """온도·습도가 STUCK2_MIN 칸 이상 같은 값일 때, 그 구간 동안 같은 공간 다른 항목이 움직인 폭(÷ 1시간 중앙값)의 최댓값.
    빈방에서는 온도가 몇 시간씩 그대로여도 다른 항목도 조용하지만, 센서 하나만 멈추면 나머지는 계속 움직인다."""
    out = {}
    for sp in spaces:
        f = empty_scores(wide.index)
        for m in PAIR_METRICS:
            if (sp, m) not in wide:
                continue
            run, grp = causal_run(wide[(sp, m)])
            moved = []
            for o in METRICS:
                if o == m or (sp, o) not in wide:
                    continue
                s = wide[(sp, o)]
                moved.append((s.groupby(grp).cummax() - s.groupby(grp).cummin()) / move.get((sp, o), 1.0))
            if moved:
                f[m] = pd.concat(moved, axis=1).max(axis=1, skipna=True).where(run >= STUCK2_MIN)
        out[sp] = f
    return out


# ---------------------------------------------------------------- 탐지기 묶음

@dataclass
class DeepModels:
    """정상 기간에서 정한 통계 (시험 기간은 보지 않음)."""

    spaces: list[str]
    tz: str
    holidays: list[str]
    gru: Any = None                       # (model, spec) 또는 None
    res_scale: dict = field(default_factory=dict)
    floor: dict = field(default_factory=dict)
    cross: dict = field(default_factory=dict)
    move: dict = field(default_factory=dict)
    _res: dict = field(default_factory=dict, repr=False)   # 프레임별 GRU 오차 캐시 (예측오차·CUSUM 이 같이 씀)

    def residuals(self, w: pd.DataFrame):
        if id(w) not in self._res:
            self._res[id(w)] = (w, gru_residuals(*self.gru, w))
        return self._res[id(w)][1]

    def detectors(self) -> dict[str, Callable[[pd.DataFrame], Scores]]:
        det: dict[str, Callable[[pd.DataFrame], Scores]] = {}
        if self.gru is not None and self.res_scale:
            det["pe"] = lambda w: pe_scores(self.residuals(w), self.res_scale, self.spaces, w.index, self.tz)
            det["cusum"] = lambda w: cusum_scores(self.residuals(w), self.res_scale, self.spaces, w.index, self.tz)
        det["floor"] = lambda w: floor_scores(w, self.spaces, self.floor, self.tz, self.holidays)
        det["cross"] = lambda w: cross_scores(w, self.spaces, self.cross)
        det["stuck2"] = lambda w: stuck2_scores(w, self.spaces, self.move)
        return det


def fit_deep(train: pd.DataFrame, spaces: list[str], cfg: dict[str, Any], gru=None) -> DeepModels:
    tz, hol = cfg.get("timezone", "Asia/Seoul"), [str(d) for d in cfg.get("holidays", [])]
    dm = DeepModels(spaces, tz, hol, gru=gru)
    if gru is not None:
        dm.res_scale = fit_residual_scale(gru_residuals(*gru, train), tz)
    dm.floor = fit_floor(train, spaces, tz, hol)
    dm.cross = fit_cross_scale(train, spaces)
    dm.move = fit_move_scale(train, spaces)
    return dm


# ---------------------------------------------------------------- 빠른 채점 (조합 선택용 — anomaly.evaluate 와 같은 기준)

def any_flags(flags: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    return {sp: fl.any(axis=1).to_numpy() for sp, fl in flags.items()}


def fault_hits(anyf: dict[str, np.ndarray], index: pd.DatetimeIndex, faults: list[Fault]) -> np.ndarray:
    out = np.zeros(len(faults), bool)
    for i, f in enumerate(faults):
        sel = (index >= f.start) & (index <= f.end + GRACE)
        out[i] = bool(anyf[f.space][sel].any())
    return out


def false_alarms(anyf: dict[str, np.ndarray], index: pd.DatetimeIndex, faults: list[Fault], days: float) -> float:
    fa = 0
    for sp, a in anyf.items():
        d = np.diff(np.r_[0, a.astype(np.int8), 0])
        starts, ends = index[np.where(d == 1)[0]], index[np.where(d == -1)[0] - 1]
        mine = [f for f in faults if f.space == sp]
        fa += sum(not any(s <= f.end + GRACE and e >= f.start for f in mine) for s, e in zip(starts, ends))
    return fa / max(days, 1e-9)


def _or(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {sp: a.copy() for sp, a in parts[0].items()}
    for p in parts[1:]:
        for sp, a in p.items():
            out[sp] |= a
    return out


def select_combo(base: dict, cand: dict[tuple[str, float], dict], faults: list[Fault], index: pd.DatetimeIndex,
                 days: float, names: list[str], limit: float = FA_LIMIT) -> tuple[tuple, pd.DataFrame]:
    """검증 기간(따로 넣은 고장)에서 '규칙 OR 오토인코더' 에 새 탐지기(꺼짐 또는 예산 하나)를 더한 모든 조합을 채점.
    base/cand 값: {"inj": 공간→불리언 배열, "clean": 공간→불리언 배열, "hits": 고장별 탐지}.
    헛경보(주입·원본 중 큰 값) ≤ limit 에서 탐지율 최대, 같으면 헛경보 적은 것, 그다음 탐지기 적은 것."""
    rows = []
    for choice in itertools.product(*[[None, *BUDGETS] for _ in names]):
        used = [(n, b) for n, b in zip(names, choice) if b is not None]
        hits = base["hits"].copy()
        for k in used:
            hits |= cand[k]["hits"]
        fa_inj = false_alarms(_or(base["inj"], *[cand[k]["inj"] for k in used]), index, faults, days)
        fa_clean = false_alarms(_or(base["clean"], *[cand[k]["clean"] for k in used]), index, [], days)
        rows.append({"combo": tuple(used), "detection_rate": round(hits.mean() * 100, 1),
                     "fa_inj": round(fa_inj, 2), "fa_clean": round(fa_clean, 2), "n_det": len(used)})
    df = pd.DataFrame(rows)
    df["fa"] = df[["fa_inj", "fa_clean"]].max(axis=1)
    ok = df[df["fa"] <= limit].sort_values(["detection_rate", "fa", "n_det"], ascending=[False, True, True])
    best = ok.iloc[0]["combo"] if len(ok) else ()
    return best, df.sort_values(["detection_rate", "fa"], ascending=[False, True])


# ---------------------------------------------------------------- 과제 안내 고장 5종 (냉방 정지·환기 정지 주입)

def usage_mask(index: pd.DatetimeIndex, tz: str, holidays: list[str]) -> np.ndarray:
    """평일(공휴일 제외) 한국 시각 9~18시 사용 시간."""
    hr = local_hours(index, tz)
    return workdays(index, tz, holidays) & (hr >= USE_HOURS[0]) & (hr < USE_HOURS[1])


def inject_usage(wide: pd.DataFrame, spaces: list[str], per_kind: int = 15, seed: int = 0, tz: str = "Asia/Seoul",
                 holidays: list[str] | None = None, busy: list[Fault] | None = None,
                 step_min: int = 5) -> tuple[pd.DataFrame, list[Fault]]:
    """사용 시간 고장 2종(cooling_off, vent_off)을 넣은 복사본과 정답 목록.
    고장 구간 전체가 한 사용 시간(평일 9~18시) 안에 들어가는 자리만 고른다. busy(이미 넣은 고장)와 T4_SEP 만큼 띄운다.
      cooling_off: 온도 += 0→amp 선형 (amp 3~5℃), 끝 다음 칸부터 원래 값
      vent_off   : CO₂ = 원본 + (1.8-1)×max(원본 - 시작값, 0) — 오른 만큼만 키우고 내려간 칸은 그대로"""
    rng = np.random.default_rng(seed)
    w = wide.copy()
    step = pd.Timedelta(minutes=step_min)
    use = usage_mask(w.index, tz, list(holidays or []))
    taken: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {sp: [] for sp in spaces}
    for f in busy or []:
        if f.space in taken:
            taken[f.space].append((f.start, f.end))
    faults: list[Fault] = []
    lo_n, hi_n = USAGE_HOURS[0] * 60 // step_min, USAGE_HOURS[1] * 60 // step_min
    for kind, metric in (("cooling_off", "temperature"), ("vent_off", "co2")):
        cand = [sp for sp in spaces if (sp, metric) in w]
        placed, tries = 0, 0
        while cand and placed < per_kind and tries < per_kind * 400:
            tries += 1
            sp = cand[rng.integers(len(cand))]
            n = int(rng.integers(lo_n, hi_n + 1))
            if len(w) <= n:
                break
            i0 = int(rng.integers(0, len(w) - n))
            if not use[i0:i0 + n].all():
                continue                          # 사용 시간 밖으로 걸치면 버림 (밤·주말·공휴일 포함)
            t0, t1 = w.index[i0], w.index[i0] + n * step
            if w.index[i0 + n - 1] != t1 - step:
                continue                          # 인덱스가 빠진 구간
            if any(t0 < e + T4_SEP and t1 > s - T4_SEP for s, e in taken[sp]):
                continue
            seg = w[(sp, metric)].iloc[i0:i0 + n]
            if seg.isna().mean() > 0.1 or seg.std() == 0:
                continue
            idx = w.index[i0:i0 + n]
            if kind == "cooling_off":
                amp = float(rng.uniform(*COOL_AMP))
                w.loc[idx, (sp, metric)] += np.linspace(0, amp, n)
            else:
                c0 = seg.iloc[0]
                if not np.isfinite(c0) or not seg.iloc[-1] - c0 >= VENT_MIN_RISE:
                    continue                      # CO₂ 가 오르는 구간이 아님
                w.loc[idx, (sp, metric)] = seg + (VENT_GAIN - 1) * (seg - c0).clip(lower=0)
            taken[sp].append((t0, t1))
            faults.append(Fault(kind, sp, metric, t0, t1))
            placed += 1
    return w, faults


def inject_task_kinds(clean: pd.DataFrame, injected: pd.DataFrame, faults: list[Fault], spaces: list[str],
                      per_kind: int, seed: int, tz: str, holidays: list[str],
                      outdoor: str | None = None) -> tuple[pd.DataFrame, list[Fault]]:
    """과제 4 시험 주입에서 spike/stuck/drift 는 그대로 두고 noise/dropout 은 원본으로 되돌린 뒤,
    실내 공간(outdoor 제외)에 cooling_off/vent_off 를 더한다. 과제 4 고장 사이는 6시간 이상이라 되돌리기가 정확하다."""
    w = injected.copy()
    keep = [f for f in faults if f.kind in T4_KINDS]
    for f in faults:
        if f.kind in T4_KINDS:
            continue
        cols = [(f.space, m) for m in METRICS if (f.space, m) in w] if f.metric == "*" else [(f.space, f.metric)]
        sel = (w.index >= f.start) & (w.index < f.end)
        w.loc[sel, cols] = clean.loc[sel, cols]
    indoor = [sp for sp in spaces if sp != outdoor]
    w, new = inject_usage(w, indoor, per_kind=per_kind, seed=seed, tz=tz, holidays=holidays, busy=keep)
    return w, keep + new


def fault_firsts(flags: dict[str, pd.DataFrame], faults: list[Fault]) -> pd.DataFrame:
    """고장마다 첫 경보: 탐지 여부(시작~끝+30분), 지연(분), 진행 중 탐지, 첫 경보 칸에 켜진 탐지기 (여럿이면 모두).
    진행 중 = 고장 구간(시작 ≤ t < 끝) 안의 경보. 단 급변 규칙은 다음 칸과의 차이도 보므로(diff(-1)) 끝에서 원래 값으로
    돌아가는 계단이 마지막 칸에 켜진다 — 마지막 칸의 급변 경보만은 빼서 그 계단만 잡은 경우를 진행 중으로 세지 않는다.
    한 칸짜리(튄 값)는 마지막 칸이 곧 시작이라 그대로 둔다."""
    rows = []
    for f in faults:
        fl = flags[f.space]
        win = fl[(fl.index >= f.start) & (fl.index <= f.end + GRACE)]
        hit = win.any(axis=1)
        first = hit.idxmax() if hit.any() else None
        by = []
        if first is not None:
            on = [c for c in win.columns if win.at[first, c]]
            by = sorted({DETECTOR_LABELS.get(c.split(":")[0], c.split(":")[0]) for c in on})
        during = win[win.index < f.end]
        if len(during) > 1:
            during = during.copy()
            during.loc[during.index[-1], [c for c in during.columns if c.startswith("spike:")]] = False
        rows.append({"kind": f.kind, "detected": first is not None,
                     "delay_min": (first - f.start).total_seconds() / 60 if first is not None else np.nan,
                     "in_progress": bool(during.any(axis=None)), "by": by})
    return pd.DataFrame(rows, columns=["kind", "detected", "delay_min", "in_progress", "by"])


def kind_table(firsts: pd.DataFrame, fa_per_day: float, kinds=T4_KINDS) -> pd.DataFrame:
    """종류별 건수·탐지율·진행 중 탐지율·지연 중앙값, 헛경보/일 (전체 공간 합, 종류와 무관한 한 값)."""
    rows = {}
    for k in (*kinds, "전체"):
        d = firsts if k == "전체" else firsts[firsts["kind"] == k]
        name = "전체" if k == "전체" else f"{k} ({T4_LABELS.get(k, k)})"
        rows[name] = {"건수": len(d),
                      "탐지율(%)": round(d["detected"].mean() * 100, 1) if len(d) else np.nan,
                      "진행 중 탐지(%)": round(d["in_progress"].mean() * 100, 1) if len(d) else np.nan,
                      "지연 중앙값(분)": d["delay_min"].median() if len(d) else np.nan,
                      "헛경보/일": round(fa_per_day, 2)}
    return pd.DataFrame.from_dict(rows, orient="index")


def caught_by_lines(firsts: pd.DataFrame, kinds=T4_KINDS) -> list[str]:
    """종류마다 한 줄: 첫 경보를 낸 탐지기별 건수와 놓친 건수."""
    out = []
    for k in kinds:
        d = firsts[firsts["kind"] == k]
        cnt = pd.Series([b for bs in d["by"] for b in bs], dtype=object).value_counts()
        parts = [f"{name} {n}" for name, n in cnt.items()]
        miss = int((~d["detected"]).astype(bool).sum()) if len(d) else 0
        late = int((d["detected"].astype(bool) & ~d["in_progress"].astype(bool)).sum()) if len(d) else 0
        out.append(f"- {k} ({T4_LABELS[k]}, {len(d)}건): " + (", ".join(parts) if parts else "첫 경보 없음")
                   + f" / 끝난 뒤에만 경보 {late} / 놓침 {miss}")
    return out


# ---------------------------------------------------------------- 실행

def load_ae(path=an.MODEL_PATH) -> an.AE | None:
    if not path.exists():
        return None
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = an.build_ae(ckpt["L"] * an.CH)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return an.AE(model, ckpt["stats"], ckpt["L"], ckpt["threshold"])


def load_gru():
    from tsinfer.models import forecast as fc

    if not fc.MODEL_PATH.exists():
        return None
    return fc.load(fc.MODEL_PATH)


def combo_name(combo: tuple) -> str:
    return " + ".join(f"{LABELS[n]}({b}/일)" for n, b in combo) or "(추가 없음)"


def run(cfg: dict[str, Any], wide: pd.DataFrame, per_kind: int = 15, epochs: int = 30, seed: int = 0,
        ae: an.AE | None = None, gru=None, log=print) -> dict[str, Any]:
    from tsinfer.data.split import split_by_time

    spaces = [sp for sp in cfg["spaces"] if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in METRICS)]
    train, valid, test = split_by_time(wide, cfg)
    ae_src = "저장된 models/anomaly_ae.pt"
    if ae is None:
        log("오토인코더 파일이 없어 과제 4 와 같은 설정으로 다시 학습합니다")
        ae = an.fit_ae(train, spaces, epochs=epochs, threads=cfg.get("model", {}).get("threads", 3))
        ae.threshold = an.pick_threshold(ae, valid, spaces)
        ae_src = f"다시 학습 ({epochs} 에폭)"
    normal = pd.concat([train, valid])
    kw = {"noise_thr": an.fit_noise_thresholds(normal, spaces), "stuck_steps": an.fit_stuck_steps(normal, spaces)}
    dm = fit_deep(train, spaces, cfg, gru=gru)
    dets = dm.detectors()
    names = list(dets)
    log(f"새 탐지기: {', '.join(names)}" + ("" if gru is not None else " (GRU 없음 → 예측 오차 탐지기 건너뜀)"))

    def days_of(df):
        return (df.index[-1] - df.index[0]) / pd.Timedelta("1D")

    vdays, tdays = days_of(valid), days_of(test)
    valid_inj, vfaults = an.inject(valid, spaces, per_kind=per_kind, seed=VALID_SEED)
    test_inj, faults = an.inject(test, spaces, per_kind=per_kind, seed=seed)
    frames = {"valid": valid, "valid_inj": valid_inj, "test": test, "test_inj": test_inj}

    # 기존 방법 (과제 4 와 같은 함수·같은 임계값)
    base_flags = {k: {m: an.method_flags(m, w, spaces, ae, **kw) for m in ("rules", "ae")} for k, w in frames.items()}

    # 새 탐지기: 점수 → 검증(원본)에서 예산별 임계값
    scores = {}
    for n, fn in dets.items():
        log(f"  점수 계산: {LABELS[n]}")
        scores[n] = {k: fn(w) for k, w in frames.items()}
    thr = {(n, b): pick_score_threshold(scores[n]["valid"], vdays, b) for n in names for b in BUDGETS}
    new_flags = {key: {k: score_flags(scores[key[0]][k], t, key[0]) for k in frames} for key, t in thr.items()}

    # 검증 기간에서 결합 조합 선택
    base_v = {"inj": _or(any_flags(base_flags["valid_inj"]["rules"]), any_flags(base_flags["valid_inj"]["ae"])),
              "clean": _or(any_flags(base_flags["valid"]["rules"]), any_flags(base_flags["valid"]["ae"]))}
    base_v["hits"] = fault_hits(base_v["inj"], valid.index, vfaults)
    cand = {}
    for key, fl in new_flags.items():
        c = {"inj": any_flags(fl["valid_inj"]), "clean": any_flags(fl["valid"])}
        c["hits"] = fault_hits(c["inj"], valid.index, vfaults)
        cand[key] = c
    thr_fa = {key: false_alarms(c["clean"], valid.index, [], vdays) for key, c in cand.items()}
    best, combos = select_combo(base_v, cand, vfaults, valid.index, vdays, names)
    log(f"선택된 조합 (검증 기간 기준): 규칙 + 오토인코더 + {combo_name(best)}")

    # 시험 기간 채점
    def merged(frame: str, parts: list[str], combo: tuple = ()) -> dict[str, pd.DataFrame]:
        out = {}
        for sp in spaces:
            dfs = [base_flags[frame][p][sp] for p in parts] + [new_flags[k][frame][sp] for k in combo]
            out[sp] = pd.concat(dfs, axis=1)
        return out

    methods: dict[str, tuple[list[str], tuple]] = {"rules": (["rules"], ()), "ae": (["ae"], ())}
    for n in names:
        methods[n] = ([], ((n, BUDGETS[0]),))
    methods["base"] = (["rules", "ae"], ())
    drift_only = tuple((n, BUDGETS[0]) for n in ("cusum", "floor", "cross") if n in names)
    methods["drift"] = ([], drift_only)
    methods["all"] = (["rules", "ae"], tuple((n, BUDGETS[0]) for n in names))
    methods["final"] = (["rules", "ae"], best)
    results, clean_fa, test_flags, clean_flags = {}, {}, {}, {}
    methods = {m: v for m, v in methods.items() if v[0] or v[1]}
    for m, (parts, combo) in methods.items():
        test_flags[m] = merged("test_inj", parts, combo)
        clean_flags[m] = merged("test", parts, combo)
        results[m] = an.evaluate(test_flags[m], faults, tdays)
        clean_fa[m] = an.evaluate(clean_flags[m], [], tdays)["false_alarms_per_day"].iloc[0]
    # 검증 기간 (조합 고를 때 본 값) — 같은 방법별
    valid_res = {m: an.evaluate(merged("valid_inj", parts, combo), vfaults, vdays) for m, (parts, combo) in methods.items()}

    # 과제 안내 고장 5종 시험 — 위에서 정한 임계값·조합 그대로, 새 프레임에 다시 적용만 한다
    outdoor = cfg.get("model", {}).get("outdoor_space")
    t4_inj, t4_faults = inject_task_kinds(test, test_inj, faults, spaces, per_kind, seed, dm.tz, dm.holidays, outdoor)
    t4_base = an.method_flags("combined", t4_inj, spaces, ae, **kw)     # 규칙 열 + 오토인코더 열
    t4_new = {k: score_flags(dets[k[0]](t4_inj), thr[k], k[0]) for k in best}
    t4_flags = {"base": t4_base,
                "final": {sp: pd.concat([t4_base[sp]] + [t4_new[k][sp] for k in best], axis=1) for sp in spaces}}
    t4 = {}
    for m, fl in t4_flags.items():
        firsts = fault_firsts(fl, t4_faults)
        t4[m] = {"firsts": firsts, "table": kind_table(firsts, false_alarms(any_flags(fl), test.index, t4_faults, tdays))}
    log("과제 안내 5종 시험: " + ", ".join(f"{k} {sum(f.kind == k for f in t4_faults)}건" for k in T4_KINDS))

    return {"t4": t4, "t4_faults": t4_faults, "seed": seed,"spaces": spaces, "ae": ae, "ae_src": ae_src, "gru": gru, "dm": dm, "names": names, "thr": thr, "thr_fa": thr_fa,
            "best": best, "combos": combos, "methods": methods, "results": results, "clean_fa": clean_fa,
            "valid_res": valid_res, "faults": faults, "vfaults": vfaults, "test_flags": test_flags,
            "clean_flags": clean_flags, "period": (test.index[0], test.index[-1]),
            "valid_period": (valid.index[0], valid.index[-1]), "tdays": tdays, "vdays": vdays,
            "kw": kw, "tz": dm.tz}


# ---------------------------------------------------------------- 보고서

METHOD_TITLES = {"rules": "규칙", "ae": "오토인코더", "pe": "예측오차", "cusum": "예측오차 누적(CUSUM)",
                 "floor": "바닥 CO₂", "cross": "공간 간 비교", "stuck2": "교차 값 고정",
                 "base": "기존 결합(규칙 OR AE)", "drift": "드리프트 전용(CUSUM OR 바닥 OR 공간 간)",
                 "all": "모두 결합(새 탐지기 0.5/일)", "final": "최종 결합(검증으로 선택)"}


def comparison_table(out: dict[str, Any]) -> pd.DataFrame:
    cols = {}
    for m in out["methods"]:
        r = out["results"][m]
        col = r["detection_rate"].copy()
        col.loc["drift 진행 중 탐지(끝 계단 전)"] = drift_in_progress(out["test_flags"][m], out["faults"])
        col.loc["지연 중앙값(분)"] = r.loc["전체", "delay_min_median"]
        col.loc["헛경보/일 (주입 시험)"] = r.loc["전체", "false_alarms_per_day"]
        col.loc["헛경보/일 (원본 시험)"] = round(float(out["clean_fa"][m]), 2)
        cols[METHOD_TITLES[m]] = col
    return pd.DataFrame(cols)


def drift_in_progress(flags: dict[str, pd.DataFrame], faults: list[Fault]) -> float:
    """drift 를 '어긋나는 동안'(시작 ≤ 경보 < 끝) 잡은 비율 %. 주입 drift 는 끝에서 원래 값으로 한 번에 돌아가
    그 계단을 급변 규칙·오토인코더가 잡으면 지연 ≈360분 '탐지'로 세어진다 — 그건 drift 를 잡은 게 아니다."""
    ds = [f for f in faults if f.kind == "drift"]
    if not ds:
        return np.nan
    hit = [bool(flags[f.space][(flags[f.space].index >= f.start) & (flags[f.space].index < f.end)].any(axis=None))
           for f in ds]
    return round(float(np.mean(hit)) * 100, 1)


def per_metric_table(out: dict[str, Any], flags_key: str = "test_flags", kind: str | None = None) -> pd.DataFrame:
    """고장 항목별 탐지율. 탐지기가 그 항목을 보지 않으면 '해당 없음'."""
    faults = [f for f in out["faults"] if kind is None or f.kind == kind]
    rows = {}
    for m in out["methods"]:
        fl = out[flags_key][m]
        index = next(iter(fl.values())).index
        hits = fault_hits(any_flags(fl), index, faults)
        col = {}
        for metric in (*METRICS, "*"):
            sel = np.array([f.metric == metric for f in faults])
            if not sel.any():
                continue
            if m in APPLIES and metric != "*" and metric not in APPLIES[m]:
                col[metric] = "해당 없음"
            else:
                col[metric] = f"{hits[sel].mean() * 100:.1f} ({int(sel.sum())}건)"
        rows[METHOD_TITLES[m]] = col
    return pd.DataFrame(rows).rename(index={"*": "전체 항목(끊김)"})


def drift_cases(out: dict[str, Any]) -> pd.DataFrame:
    """드리프트 고장 하나하나: 한국 시각, 새벽(0~6시) 포함 여부, 방법별 첫 경보까지 분."""
    tz = out["tz"]
    rows = []
    for f in [f for f in out["faults"] if f.kind == "drift"]:
        s, e = f.start.tz_convert(tz), f.end.tz_convert(tz)
        hours = pd.date_range(s, e, freq="5min").hour
        row = {"공간": f.space, "항목": f.metric, "시작(KST)": f"{s:%m-%d %H:%M}",
               "새벽 포함": "예" if ((hours >= 0) & (hours < 6)).any() else "아니오"}
        for m in ("rules", "ae", *[n for n in out["names"]], "final"):
            fl = out["test_flags"][m][f.space]
            win = fl[(fl.index >= f.start) & (fl.index <= f.end + GRACE)].any(axis=1)
            row[METHOD_TITLES[m].split("(")[0]] = int((win.idxmax() - f.start).total_seconds() // 60) if win.any() else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def build_report(out: dict[str, Any], cfg: dict[str, Any], per_kind: int) -> list[str]:
    t0, t1 = out["period"]
    v0, v1 = out["valid_period"]
    thr_rows = [{"탐지기": LABELS[n], "예산(회/일)": b, "임계값": round(t, 3) if np.isfinite(t) else "∞",
                 "검증(원본) 경보/일": round(out["thr_fa"][(n, b)], 2)} for (n, b), t in out["thr"].items()]
    combos = out["combos"].copy()
    combos["combo"] = combos["combo"].map(combo_name)
    best_row = combos[combos["combo"] == combo_name(out["best"])]
    clean_by, clean_ev = an.alarm_causes(out["clean_flags"]["final"], out["tdays"])
    gru_txt = "없음 (models/forecast_gru.pt 없음 → 예측오차·CUSUM 건너뜀)"
    if out["gru"] is not None:
        spec = out["gru"][1]
        gru_txt = (f"models/forecast_gru.pt (입력 {spec.lookback * spec.step_minutes}분, 대상 {', '.join(spec.spaces)}), "
                   f"지평 {[h * spec.step_minutes for h in HORIZONS]}분")
    lines = [
        "# 과제 4 심화. 예측 오차 탐지와 영점 틀어짐(drift) 전용 탐지",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터, 학습(정상) ~{cfg['split']['train_end']}, 검증 {v0:%Y-%m-%d} ~ {v1:%Y-%m-%d}, "
        f"시험 {t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} (UTC) — 과제 4 와 같은 분할",
        f"- 대상 공간: {', '.join(out['spaces'])} / 고장 주입: 과제 4 와 같음 (시험 seed 0, 종류별 {per_kind}건, "
        f"drift = 6시간 동안 CO₂ +250ppm / 온도 +3℃ / 습도 +12% 선형, 70% 는 CO₂)",
        f"- 규칙·오토인코더: anomaly.py 함수 그대로 (오토인코더 {out['ae_src']}, 임계값 {out['ae'].threshold:.3f})",
        f"- GRU: {gru_txt}. 오차 단위 = 학습 기간 한국 시각별 평균 절대 오차",
        f"- CUSUM: 지평 평균 정규화 오차, 허용치 k={CUSUM_K}, 한 칸 상한 ±{Z_CLIP}",
        f"- 바닥 CO₂: 최근 {FLOOR_WIN * 5}분 최솟값 vs 학습 기간 (평일/휴일 × 한국 시각) {FLOOR_Q:.0%} 분위수 (앞뒤 1시간 최댓값)",
        f"- 공간 간 비교: 온도·습도, (내 값 - 다른 공간 중앙값)의 {CROSS_SMOOTH * 5}분 중앙값을 {CROSS_LAG * 5 // 60}시간 전과 비교, "
        "학습 기간 표준편차로 나눔, 같은 시점 가장 큰 공간만",
        f"- 교차 값 고정: 온도·습도가 {STUCK2_MIN * 5}분 이상 같은 값일 때 같은 공간 다른 항목이 움직인 폭 (÷ 학습 기간 1시간 폭 중앙값)",
        f"- 임계값: 탐지기마다 검증 기간(원본) 경보 사건 ≤ {', '.join(map(str, BUDGETS))} 회/일 중 가장 낮은 값",
        f"- 결합 선택: 검증 기간에 따로 넣은 고장(seed {VALID_SEED}, 종류별 {per_kind}건 시도, 실제 {len(out['vfaults'])}건)으로 "
        f"규칙 OR AE + 새 탐지기(꺼짐 / 예산 {BUDGETS}) 모든 조합을 채점, 검증 헛경보(주입·원본 중 큰 값) ≤ {FA_LIMIT}회/일 안에서 탐지율 최대. "
        "시험 기간은 임계값·조합 선택에 쓰지 않음",
        f"- 탐지 = 고장 시작~끝+{int(GRACE.total_seconds() // 60)}분 안의 경보, 헛경보 = 고장과 겹치지 않는 경보 사건 (전체 공간 합 / 일)",
        "",
        "## 탐지기별 임계값 (검증 기간에서 결정)",
        "",
        pd.DataFrame(thr_rows).to_markdown(index=False),
        "",
        "## 비교 (시험 기간, 탐지율 %) — 새 탐지기 단독은 예산 0.5회/일 임계값",
        "",
        comparison_table(out).to_markdown(),
        "",
        "### 고장 항목별 탐지율 (전체 종류, %) — GRU 는 습도를 예측하지 않아 예측오차·CUSUM 의 습도는 해당 없음",
        "",
        per_metric_table(out).to_markdown(),
        "",
        "### drift 만, 항목별 탐지율 (%)",
        "",
        per_metric_table(out, kind="drift").to_markdown(),
        "",
        "### drift 고장별 첫 경보까지 (분, '-' = 놓침)",
        "",
        drift_cases(out).to_markdown(index=False),
        "",
        "## 결합 조합 선택 (검증 기간, 시험 기간 미사용)",
        "",
        f"- 선택: **규칙 + 오토인코더 + {combo_name(out['best'])}**",
        "",
        best_row.drop(columns=["n_det"]).to_markdown(index=False) if len(best_row) else "(조건을 만족하는 조합 없음)",
        "",
        "### 검증 기간 상위 조합 10개 (헛경보 상한 무시, 참고)",
        "",
        combos.drop(columns=["n_det"]).head(10).to_markdown(index=False),
        "",
        "### 최종 결합 — 시험 기간 상세",
        "",
        out["results"]["final"].round(1).to_markdown(),
        "",
        "### 최종 결합 — 검증 기간 상세 (선택에 쓴 값)",
        "",
        out["valid_res"]["final"].round(1).to_markdown(),
        "",
        "## 원본 시험 기간 경보 (최종 결합, 원인별)",
        "",
        clean_by.to_markdown() if len(clean_by) else "(경보 없음)",
        "",
        clean_ev.to_markdown(index=False) if len(clean_ev) else "",
        "",
        *task_kinds_section(out, cfg, per_kind),
        "## 설계상 한계",
        "- 주입 drift 는 끝에서 원래 값으로 한 번에 돌아간다. 이 계단을 급변 규칙·오토인코더가 잡으면 지연 ≈360분으로 '탐지'에 "
        "들어가므로, 비교표의 'drift 진행 중 탐지' 행(경보가 어긋나는 동안 켜진 비율)을 함께 본다.",
        "- 바닥 CO₂ 는 방이 비어야 효과가 있다. 사람이 있는 낮에 시작한 drift 는 바닥이 원래 높아 묻히고, "
        "평가 창(시작~6시간+30분) 안에 빈 시간이 없으면 다음 새벽까지 기다려야 잡힌다 → 그 경우 평가에서는 놓침으로 센다.",
        "- 외기 CO₂ 센서가 없어 '외기와 비슷해야 한다'를 직접 쓰지 못하고, 학습 기간 바닥 분포로 대신한다. "
        "계절·환기 설정이 바뀌면 바닥 상한을 다시 학습해야 한다.",
        "- 예측오차·CUSUM 은 GRU 가 예측하는 CO₂·온도만 본다. 입력 창이 고장 값을 보기 시작하면 예측도 따라가므로 "
        "느린 drift 는 한 칸 오차가 작다 — 그래서 누적(CUSUM)을 함께 본다.",
        "- 공간 간 비교는 공간이 3개 이상이어야 어느 센서가 어긋났는지 가릴 수 있다 (2개면 서로 같은 크기로 보임).",
        "- 냉방 정지·환기 정지는 센서 고장이 아니라 설비 고장을 흉내 낸 합성 주입이다. 냉방 정지는 끝에서 원래 온도로 한 번에 "
        "돌아가고 환기 정지도 끝에서 키운 상승분이 사라지므로, 그 계단을 급변 규칙이 잡으면 '진행 중 탐지'가 아닌 탐지로 "
        "세어진다. 실제 냉방·환기 정지 사례(설비 기록)로 확인하지 않았다.",
        "",
        "## 해석 (직접 작성)",
        "- 예측오차 탐지기 vs 오토인코더 (어떤 고장을 누가 더 잘 잡나, 이유):",
        "- drift 를 잡은 사례 / 놓친 사례의 공통점 (낮·새벽, 항목):",
        "- 과제 안내 5종: 냉방 정지·환기 정지를 누가 잡았나, 놓친 이유 (검증 조합 선택에 없던 종류):",
        "- 최종 결합을 운영에 쓸 때 헛경보와 탐지율의 균형:",
        "",
    ]
    return lines


def task_kinds_section(out: dict[str, Any], cfg: dict[str, Any], per_kind: int) -> list[str]:
    """과제 안내 고장 5종 시험 절. 기존 결합 / 최종 결합 각각 종류별 표와 '누가 잡았나' 줄."""
    if "t4" not in out:
        return []
    t4f = out["t4_faults"]
    counts = ", ".join(f"{k} {sum(f.kind == k for f in t4f)}" for k in T4_KINDS)
    lines = [
        "## 과제 안내 고장 5종 시험 (시험 기간)",
        "",
        "과제 안내의 고장 5종(튄 값·값 고정·영점 틀어짐·냉방 정지·환기 정지)으로 다시 채점한다. "
        "임계값·결합 조합은 위에서 검증 기간으로 정한 것을 그대로 쓰고 시험 기간으로 다시 고르지 않는다 "
        "(냉방 정지·환기 정지는 조합을 고를 때 없던 종류다).",
        "",
        "- spike·stuck·drift: 과제 4 시험 주입(seed 0)의 같은 고장 그대로, noise·dropout 자리는 원본으로 되돌림",
        f"- cooling_off(냉방 정지): 평일 {USE_HOURS[0]}~{USE_HOURS[1]}시(한국 시각, 공휴일 제외) 안에서 "
        f"{USAGE_HOURS[0]}~{USAGE_HOURS[1]}시간 동안 온도 +{COOL_AMP[0]:g}~{COOL_AMP[1]:g}℃ 선형, 끝에서 원래 값",
        f"- vent_off(환기 정지): 같은 사용 시간 중 원본 CO₂ 가 구간 끝에서 시작보다 {VENT_MIN_RISE:g}ppm 이상 오른 "
        f"{USAGE_HOURS[0]}~{USAGE_HOURS[1]}시간, 시작 시점 대비 상승분 ×{VENT_GAIN:g}",
        f"- 냉방·환기 정지 대상: 실내 공간만 ({cfg.get('model', {}).get('outdoor_space') or '외기 공간'} 제외), "
        f"seed {out.get('seed', 0)}, 종류별 {per_kind}건 시도, 다른 고장과 {int(T4_SEP.total_seconds() // 3600)}시간 이상 간격 — 실제 {counts}건",
        "- 진행 중 탐지 = 고장 구간(시작~끝) 안에 경보가 켜진 비율. 급변 규칙은 다음 칸 차이도 봐서 끝에서 원래 값으로 "
        "돌아가는 계단이 마지막 칸에 켜지므로 마지막 칸의 급변 경보는 뺀다 (한 칸짜리 튄 값은 그대로), "
        "헛경보/일 = 이 주입 시험에서 고장과 겹치지 않는 경보 사건 (전체 공간 합)",
        "",
    ]
    for m in ("base", "final"):
        title = METHOD_TITLES[m] + (f": 규칙 + 오토인코더 + {combo_name(out['best'])}" if m == "final" else "")
        res = out["t4"][m]
        lines += [f"### {title}", "", res["table"].to_markdown(), "",
                  "누가 잡았나 (원인별 첫 경보 건수, 같은 칸에 여럿이 켜지면 각각 셈):", "",
                  *caught_by_lines(res["firsts"]), ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-kind", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30, help="저장된 오토인코더가 없을 때만 학습에 쓰는 에폭")
    args = ap.parse_args()

    import torch
    from tsinfer.data.preprocess import clean, load_raw, to_wide

    cfg = load_yaml("default.yaml")
    torch.set_num_threads(cfg.get("model", {}).get("threads", 3))
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    gru = load_gru()
    if gru is None:
        print("models/forecast_gru.pt 가 없어 예측 오차 탐지기를 건너뜁니다 (python -m tsinfer.models.forecast 먼저 실행)")
    out = run(cfg, wide, args.per_kind, args.epochs, ae=load_ae(), gru=gru)
    lines = build_report(out, cfg, args.per_kind)
    path = ROOT / "reports" / "04b_anomaly_deep.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
