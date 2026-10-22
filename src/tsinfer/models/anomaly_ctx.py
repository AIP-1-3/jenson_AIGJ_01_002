"""과제 4 개선(C): 사용 시간 맥락을 아는 탐지기 + 과제 안내 고장 5종으로 검증 선택.

    python -m tsinfer.models.anomaly_ctx     # → reports/04c_anomaly_ctx.md (models/forecast_gru.pt, models/anomaly_ae.pt 재사용)

04b 와 달라진 점
1. 선택 목표를 과제 안내 고장 5종(튄 값·값 고정·영점 틀어짐·냉방 정지·환기 정지)에 맞춘다.
   04b 는 검증 기간에 anomaly.py 고장(잡음·끊김 포함)을 넣고 골라서, 채점에 쓰는 냉방·환기 정지를 고를 때 보지 못했다.
   여기서는 검증 기간에 5종을 넣고(seed 1, 시험 seed 0 과 다름) 그 탐지율로 고른다.
2. 사용 시간 맥락(평일 9~18시, 공휴일 제외)을 아는 탐지기를 더한다. 빈방은 값이 몇 시간씩 그대로인 게 정상이고,
   사람이 있을 때는 값이 계속 움직이는 게 정상이다.
     stuck_ctx  : 맥락 값 고정 — 지금까지 같은 값이 이어진 길이 ÷ 정상 기간의 (공간·항목·사용/빈 시간)별 99.9% 길이
     stuck2_use : 교차 값 고정(04b stuck2)을 사용 시간에만 — 빈 시간의 교차 값 고정 경보를 끈다
     pe_long    : 예측오차 30·60분 — 환기 정지처럼 한 시간에 걸쳐 쌓이는 어긋남 (04b 는 5·15분만 봄)
     high       : 고농도 CO₂ — 지금 값 − 정상 기간 (평일/휴일 × 한국 시각)별 99% 값
3. 조합 후보는 04b 탐지기 5개 + 새 탐지기 4개, 각각 꺼짐 / 예산 0.25 / 0.1 회/일 (4개를 켜도 1회/일 안쪽).
   검증 헛경보(주입·원본 중 큰 값) ≤ 1회/일 안에서 5종 탐지율이 가장 높은 조합 → 같으면 헛경보 적은 것 → 탐지기 적은 것.

공정성: 임계값·조합은 검증 기간에서만 정한다. 다만 이 개선은 04b 시험 결과(빈 회의실 교차 값 고정 헛경보)를 본 뒤
설계했다는 점을 보고서에 밝힌다. 10-21 이후 새로 쌓인 기간에서 다시 확인하는 것이 가장 깨끗한 검증이다.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Any, Callable

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import anomaly as an
from tsinfer.models import anomaly_deep as ad
from tsinfer.models.anomaly import GRACE, METRICS, Fault

REPORT = ROOT / "reports" / "04c_anomaly_ctx.md"
BUDGETS = (0.25, 0.1)
FA_LIMIT = 1.0
STUCK_Q = 0.999
HIGH_Q = 0.99
LONG_HORIZONS = (6, 12)        # 30·60분
REF_COMBO = (("pe", 0.25), ("stuck2", 0.25))    # 04b 최종 결합
LABELS = {**ad.LABELS, "stuck_ctx": "맥락 값 고정", "stuck2_use": "교차 값 고정(사용 시간만)",
          "pe_long": "예측오차 30·60분", "high": "고농도 CO₂"}


# ---------------------------------------------------------------- 새 탐지기

def fit_flat_ctx(normal: pd.DataFrame, spaces: list[str], tz: str, hol: list[str],
                 q: float = STUCK_Q) -> dict[tuple[str, str, int], float]:
    """(공간, 항목, 사용 1 / 빈 0) → 정상 기간에 같은 값이 이어진 길이(칸)의 q 분위수. 칸이 없으면 전체 분위수."""
    use = ad.usage_mask(normal.index, tz, hol)
    out = {}
    for sp in spaces:
        for m in METRICS:
            if (sp, m) not in normal:
                continue
            run = an.flat_run_steps(normal[(sp, m)])
            pos = (run > 0).to_numpy()
            allq = run[pos].quantile(q) if pos.any() else 1.0
            for c, sel in ((1, use), (0, ~use)):
                r = run[pos & sel]
                v = r.quantile(q) if len(r) else allq
                out[(sp, m, c)] = float(max(v if np.isfinite(v) else allq, 1.0))
    return out


def stuck_ctx_scores(wide: pd.DataFrame, spaces: list[str], table: dict, tz: str, hol: list[str]) -> ad.Scores:
    use = ad.usage_mask(wide.index, tz, hol)
    out = {}
    for sp in spaces:
        f = ad.empty_scores(wide.index)
        for m in METRICS:
            if (sp, m, 1) not in table:
                continue
            run, _ = ad.causal_run(wide[(sp, m)])
            f[m] = run.to_numpy() / np.where(use, table[(sp, m, 1)], table[(sp, m, 0)])
        out[sp] = f
    return out


def stuck2_use_scores(wide: pd.DataFrame, spaces: list[str], move: dict, tz: str, hol: list[str]) -> ad.Scores:
    use = ad.usage_mask(wide.index, tz, hol)
    return {sp: df.where(np.broadcast_to(use[:, None], df.shape)) for sp, df in ad.stuck2_scores(wide, spaces, move).items()}


def long_scores(res: dict, scale: dict, spaces: list[str], index: pd.DatetimeIndex, tz: str) -> ad.Scores:
    """30·60분 예측오차 |정규화 오차| 최댓값. 단위는 04b 예측오차와 같은 (한국 시각별 평균 절대 오차)."""
    hr = ad.local_hours(index, tz)
    out = {}
    for sp in spaces:
        f = ad.empty_scores(index)
        for m in ad.TARGETS:
            zs = [res[(sp, m, h)] / scale[(sp, m, h)][hr] for h in LONG_HORIZONS
                  if (sp, m, h) in res and (sp, m, h) in scale]
            if zs:
                f[m] = pd.concat(zs, axis=1).abs().max(axis=1, skipna=True)
        out[sp] = f
    return out


def fit_high(train: pd.DataFrame, spaces: list[str], tz: str, hol: list[str]) -> dict[str, np.ndarray]:
    """공간별 (휴일0/평일1 × 한국 시각) CO₂ 99% 값. 앞뒤 1시간과 최댓값 (fit_floor 와 같은 방식)."""
    out = {}
    for sp in spaces:
        if (sp, "co2") not in train:
            continue
        s = train[(sp, "co2")]
        allq = s.quantile(HIGH_Q)
        if not np.isfinite(allq):
            continue
        q = s.groupby([ad.workdays(s.index, tz, hol).astype(int), ad.local_hours(s.index, tz)]).quantile(HIGH_Q)
        tab = np.full((2, 24), float(allq))
        for (w, h), v in q.items():
            if np.isfinite(v):
                tab[int(w), int(h)] = v
        out[sp] = np.maximum.reduce([np.roll(tab, 1, axis=1), tab, np.roll(tab, -1, axis=1)])
    return out


def high_scores(wide: pd.DataFrame, spaces: list[str], table: dict, tz: str, hol: list[str]) -> ad.Scores:
    wd = ad.workdays(wide.index, tz, hol).astype(int)
    hr = ad.local_hours(wide.index, tz)
    out = {}
    for sp in spaces:
        f = ad.empty_scores(wide.index)
        if sp in table:
            f["co2"] = wide[(sp, "co2")].to_numpy() - table[sp][wd, hr]
        out[sp] = f
    return out


def detectors(dm: ad.DeepModels, train: pd.DataFrame, gru=None) -> dict[str, Callable[[pd.DataFrame], ad.Scores]]:
    """04b 탐지기 + 새 탐지기 4개. 통계는 학습(정상) 기간에서만."""
    tz, hol, spaces = dm.tz, dm.holidays, dm.spaces
    det = dm.detectors()
    flat = fit_flat_ctx(train, spaces, tz, hol)
    det["stuck_ctx"] = lambda w: stuck_ctx_scores(w, spaces, flat, tz, hol)
    det["stuck2_use"] = lambda w: stuck2_use_scores(w, spaces, dm.move, tz, hol)
    if gru is not None:
        scale = ad.fit_residual_scale(ad.gru_residuals(*gru, train, horizons=LONG_HORIZONS), tz)
        det["pe_long"] = lambda w: long_scores(ad.gru_residuals(*gru, w, horizons=LONG_HORIZONS), scale, spaces,
                                               w.index, tz)
    high = fit_high(train, spaces, tz, hol)
    det["high"] = lambda w: high_scores(w, spaces, high, tz, hol)
    return det


# ---------------------------------------------------------------- 빠른 조합 채점

class Scorer:
    """공간별 불리언 배열 → 고장 탐지 여부와 헛경보 사건 수. 조합 수만 번 부르므로 numpy 로만."""

    def __init__(self, index: pd.DatetimeIndex, faults: list[Fault], days: float):
        self.days = days
        self.faults = faults
        self.win = {}
        for sp in {f.space for f in faults}:
            mine = [f for f in faults if f.space == sp]
            s = index.searchsorted([f.start for f in mine])
            e = index.searchsorted([f.end + GRACE for f in mine], side="right") - 1
            self.win[sp] = (np.asarray(s), np.asarray(e))
        self.fault_pos = [(f.space, int(index.searchsorted(f.start)),
                           int(index.searchsorted(f.end + GRACE, side="right"))) for f in faults]

    def hits(self, anyf: dict[str, np.ndarray]) -> np.ndarray:
        return np.array([bool(anyf[sp][a:b].any()) for sp, a, b in self.fault_pos], bool)

    def fa(self, anyf: dict[str, np.ndarray]) -> float:
        n = 0
        for sp, a in anyf.items():
            d = np.diff(np.r_[0, a.astype(np.int8), 0])
            st, en = np.where(d == 1)[0], np.where(d == -1)[0] - 1
            if not len(st):
                continue
            if sp in self.win:
                fs, fe = self.win[sp]
                over = (st[:, None] <= fe[None, :]) & (en[:, None] >= fs[None, :])
                n += int((~over.any(axis=1)).sum())
            else:
                n += len(st)
        return n / max(self.days, 1e-9)


def select(base: dict, cand: dict, names: list[str], sc_inj: Scorer, sc_clean: Scorer,
           limit: float = FA_LIMIT) -> tuple[tuple, pd.DataFrame]:
    """base/cand 값: {"inj": 공간→배열, "clean": 공간→배열}. 모든 조합을 채점한다."""
    base_hits = sc_inj.hits(base["inj"])
    cand_hits = {k: sc_inj.hits(c["inj"]) for k, c in cand.items()}
    rows = []
    for choice in itertools.product(*[[None, *BUDGETS] for _ in names]):
        used = tuple((n, b) for n, b in zip(names, choice) if b is not None)
        hits = base_hits.copy()
        inj = {sp: a.copy() for sp, a in base["inj"].items()}
        clean = {sp: a.copy() for sp, a in base["clean"].items()}
        for k in used:
            hits |= cand_hits[k]
            for sp in inj:
                inj[sp] |= cand[k]["inj"][sp]
                clean[sp] |= cand[k]["clean"][sp]
        fa_inj, fa_clean = sc_inj.fa(inj), sc_clean.fa(clean)
        rows.append({"combo": used, "detection_rate": round(hits.mean() * 100, 1), "fa_inj": round(fa_inj, 2),
                     "fa_clean": round(fa_clean, 2), "fa": round(max(fa_inj, fa_clean), 2), "n_det": len(used)})
    df = pd.DataFrame(rows).sort_values(["detection_rate", "fa", "n_det"], ascending=[False, True, True])
    ok = df[df["fa"] <= limit]
    return (ok.iloc[0]["combo"] if len(ok) else ()), df


def combo_name(combo: tuple) -> str:
    return " + ".join(f"{LABELS[n]}({b}/일)" for n, b in combo) or "(추가 없음)"


# ---------------------------------------------------------------- 실행

def run(cfg: dict[str, Any], wide: pd.DataFrame, ae: an.AE, gru=None, per_kind: int = 15, seed: int = 0,
        log=print) -> dict[str, Any]:
    from tsinfer.data.split import split_by_time

    spaces = [sp for sp in cfg["spaces"] if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in METRICS)]
    train, valid, test = split_by_time(wide, cfg)
    normal = pd.concat([train, valid])
    kw = {"noise_thr": an.fit_noise_thresholds(normal, spaces), "stuck_steps": an.fit_stuck_steps(normal, spaces)}
    dm = ad.fit_deep(train, spaces, cfg, gru=gru)
    # 새 탐지기 통계는 학습 기간만으로 (검증 기간은 임계값·조합 고르기에 쓴다)
    dets = detectors(dm, train, gru)
    names = list(dets)
    outdoor = cfg.get("model", {}).get("outdoor_space")

    def days_of(df):
        return (df.index[-1] - df.index[0]) / pd.Timedelta("1D")

    vdays, tdays = days_of(valid), days_of(test)
    v_inj, v_f = an.inject(valid, spaces, per_kind=per_kind, seed=ad.VALID_SEED)
    v5, v5_faults = ad.inject_task_kinds(valid, v_inj, v_f, spaces, per_kind, ad.VALID_SEED, dm.tz, dm.holidays,
                                         outdoor)
    t_inj, t_faults = an.inject(test, spaces, per_kind=per_kind, seed=seed)
    t5, t5_faults = ad.inject_task_kinds(test, t_inj, t_faults, spaces, per_kind, seed, dm.tz, dm.holidays, outdoor)
    frames = {"valid": valid, "valid5": v5, "test": test, "test5": t5, "test_inj": t_inj}
    log("검증 5종: " + ", ".join(f"{k} {sum(f.kind == k for f in v5_faults)}" for k in ad.T4_KINDS))

    base = {k: an.method_flags("combined", w, spaces, ae, **kw) for k, w in frames.items()}
    scores = {}
    for n, fn in dets.items():
        log(f"  점수 계산: {LABELS[n]}")
        scores[n] = {k: fn(w) for k, w in frames.items()}
    thr = {(n, b): ad.pick_score_threshold(scores[n]["valid"], vdays, b) for n in names for b in BUDGETS}
    flags = {key: {k: ad.score_flags(scores[key[0]][k], t, key[0]) for k in frames} for key, t in thr.items()}

    sc_inj, sc_clean = Scorer(valid.index, v5_faults, vdays), Scorer(valid.index, [], vdays)
    base_v = {"inj": ad.any_flags(base["valid5"]), "clean": ad.any_flags(base["valid"])}
    cand = {key: {"inj": ad.any_flags(fl["valid5"]), "clean": ad.any_flags(fl["valid"])} for key, fl in flags.items()}
    best, combos = select(base_v, cand, names, sc_inj, sc_clean)
    log(f"선택 (검증 5종): 규칙 + 오토인코더 + {combo_name(best)}")

    def merged(frame: str, combo: tuple) -> dict[str, pd.DataFrame]:
        return {sp: pd.concat([base[frame][sp]] + [flags[k][frame][sp] for k in combo], axis=1) for sp in spaces}

    ref = tuple(k for k in REF_COMBO if k in flags)
    out = {"spaces": spaces, "names": names, "thr": thr, "best": best, "ref": ref, "combos": combos,
           "v5_faults": v5_faults, "t5_faults": t5_faults, "t_faults": t_faults, "tdays": tdays, "vdays": vdays,
           "period": (test.index[0], test.index[-1]), "valid_period": (valid.index[0], valid.index[-1]),
           "thr_fa": {key: sc_clean.fa(c["clean"]) for key, c in cand.items()}, "tz": dm.tz, "gru": gru,
           "t5": {}, "t_inj": {}, "clean_fa": {}, "valid5": {}}
    for m, combo in (("base", ()), ("ref", ref), ("final", best)):
        fl5 = merged("test5", combo)
        firsts = ad.fault_firsts(fl5, t5_faults)
        out["t5"][m] = {"firsts": firsts,
                        "table": ad.kind_table(firsts, ad.false_alarms(ad.any_flags(fl5), test.index, t5_faults, tdays))}
        out["t_inj"][m] = an.evaluate(merged("test_inj", combo), t_faults, tdays)
        clean = merged("test", combo)
        out["clean_fa"][m] = ad.false_alarms(ad.any_flags(clean), test.index, [], tdays)
        if m == "final":
            out["clean_causes"] = an.alarm_causes(clean, tdays)
        vf = merged("valid5", combo)
        out["valid5"][m] = ad.kind_table(ad.fault_firsts(vf, v5_faults),
                                         ad.false_alarms(ad.any_flags(vf), valid.index, v5_faults, vdays))
    return out


# ---------------------------------------------------------------- 보고서

TITLES = {"base": "기존 결합(규칙 OR AE)", "ref": "04b 최종 결합", "final": "04c 최종 결합(검증 5종으로 선택)"}


def summary_table(out: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for m in ("base", "ref", "final"):
        t5 = out["t5"][m]["table"]
        ti = out["t_inj"][m]
        rows.append({"방법": TITLES[m],
                     "5종 탐지율(%)": t5.loc["전체", "탐지율(%)"], "5종 진행 중 탐지(%)": t5.loc["전체", "진행 중 탐지(%)"],
                     "5종 헛경보/일": t5.loc["전체", "헛경보/일"],
                     "원본 시험 헛경보/일": round(out["clean_fa"][m], 2),
                     "anomaly.py 고장 탐지율(%)": ti.loc["전체", "detection_rate"],
                     "anomaly.py 헛경보/일": ti.loc["전체", "false_alarms_per_day"],
                     "기준 통과(≥80%, ≤1회/일)": bool(t5.loc["전체", "탐지율(%)"] >= 80
                                                   and max(t5.loc["전체", "헛경보/일"], out["clean_fa"][m]) <= 1.0)})
    return pd.DataFrame(rows)


def build_report(out: dict[str, Any], per_kind: int) -> list[str]:
    t0, t1 = out["period"]
    v0, v1 = out["valid_period"]
    combos = out["combos"].head(10).copy()
    combos["combo"] = combos["combo"].map(combo_name)
    thr_rows = [{"탐지기": LABELS[n], "예산(회/일)": b, "임계값": round(t, 3) if np.isfinite(t) else "∞",
                 "검증(원본) 경보/일": round(out["thr_fa"][(n, b)], 2)} for (n, b), t in out["thr"].items()]
    by, ev = out["clean_causes"]
    v5 = out["v5_faults"]
    lines = [
        "# 과제 4 개선(C). 사용 시간 맥락 탐지기 + 과제 안내 고장 5종으로 검증 선택",
        "",
        "## 측정 조건",
        f"- 검증 {v0:%Y-%m-%d} ~ {v1:%Y-%m-%d}, 시험 {t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} (UTC), 04b 와 같은 분할·같은 시험 주입(seed 0)",
        f"- 선택용 검증 주입: 과제 안내 5종, seed {ad.VALID_SEED}, 종류별 {per_kind}건 시도 — 실제 "
        + ", ".join(f"{k} {sum(f.kind == k for f in v5)}" for k in ad.T4_KINDS),
        "- 사용 시간 = 평일 9~18시(한국 시각, 공휴일 제외). 맥락 값 고정 표·고농도 표·예측오차 단위는 학습 기간만으로 계산",
        f"- 조합 후보: 탐지기 {len(out['names'])}개 × (꺼짐 / {' / '.join(map(str, BUDGETS))} 회/일) = "
        f"{len(out['combos']):,}개, 검증 헛경보(주입·원본 중 큰 값) ≤ {FA_LIMIT}회/일 안에서 탐지율 최대",
        "- 공정성: 임계값·조합은 검증 기간에서만 정했다. 다만 이 개선은 04b 시험 결과(빈 회의실의 교차 값 고정 헛경보)를 "
        "본 뒤 설계했다. 10-21 이후 새로 쌓인 기간에서 다시 확인하는 것이 가장 깨끗한 검증이다.",
        "",
        "## 1. 요약 (시험 기간)",
        "",
        summary_table(out).to_markdown(index=False),
        "",
        f"- 04c 최종 결합: **규칙 + 오토인코더 + {combo_name(out['best'])}**",
        f"- 04b 최종 결합: 규칙 + 오토인코더 + {combo_name(out['ref'])}",
        "",
        "## 2. 과제 안내 5종, 종류별 (시험 기간)",
        "",
    ]
    for m in ("ref", "final"):
        res = out["t5"][m]
        lines += [f"### {TITLES[m]}", "", res["table"].to_markdown(), "",
                  *ad.caught_by_lines(res["firsts"]), ""]
    lines += ["## 3. anomaly.py 고장 5종 (튐·고정·영점·잡음·끊김), 시험 기간 — 04b 와 같은 채점", "",
              out["t_inj"]["final"].to_markdown(), "",
              "## 4. 원본 시험 기간 경보 원인 (04c 최종 결합)", "", by.to_markdown(index=False) if len(by) else "(경보 없음)", "",
              "## 5. 조합 선택 (검증 기간, 상위 10개)", "", combos.to_markdown(index=False), "",
              "### 선택된 조합의 검증 기간 5종 상세", "", out["valid5"]["final"].to_markdown(), "",
              "## 6. 탐지기별 임계값 (검증 기간 원본에서 예산으로 결정)", "",
              pd.DataFrame(thr_rows).to_markdown(index=False), "",
              "## 해석 (직접 작성)",
              "- 사용 시간 맥락을 넣었을 때 헛경보가 어떻게 바뀌었나 (어느 공간·어느 시간대):",
              "- 환기 정지·값 고정에서 새로 잡힌 것과 여전히 놓친 것의 공통점:",
              "- 운영에서 이 조합을 쓴다면 받는 사람 입장에서 무엇이 좋아졌나:", ""]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-kind", type=int, default=15)
    args = ap.parse_args()

    import torch

    from tsinfer.data.preprocess import clean, load_raw, to_wide

    cfg = load_yaml("default.yaml")
    torch.set_num_threads(cfg.get("model", {}).get("threads", 3))
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    ae = ad.load_ae()
    if ae is None:
        print("models/anomaly_ae.pt 가 없습니다. python -m tsinfer.models.anomaly 를 먼저 실행하세요.")
        return 1
    out = run(cfg, wide, ae, gru=ad.load_gru(), per_kind=args.per_kind)
    lines = build_report(out, args.per_kind)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
