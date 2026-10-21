"""과제 6 심화: GRU·Chronos 앙상블 + Chronos-2 공변량. pretrained.py 의 --ensemble 이 부른다.

    ~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.pretrained --ensemble
    # → reports/06b_pretrained_deep.md (기본 보고서 06_pretrained.md 는 건드리지 않음)

(b) 앙상블: 같은 시험 시점에서 GRU 와 Chronos 중앙값의 평균(0.5 고정)·가중평균.
    가중치는 검증 기간 시점에서 항목·지평별로 MAE 가 가장 작은 값(0~1, 0.05 간격). 시험 시점은 쓰지 않는다.
(a) 공변량: chronos 패키지에 Chronos2Pipeline 이 있고 모델(amazon/chronos-2)을 받을 수 있으면
    단변량 / +온습도·일정 을 같은 시점에서 비교. import·다운로드 실패면 표에 사유를 남기고 건너뛴다.
"""
from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import load_yaml
from tsinfer.models import pretrained as pt

WEIGHT_GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
OUT_NAME = "06b_pretrained_deep.md"


# ---------------------------------------------------------------- (b) 앙상블

def collect(wide: pd.DataFrame, points: pd.DatetimeIndex, spaces: list[str], predict_fn, ctx_len: int, H: int,
            gru=None) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """(공간, 항목) → y (N,H) 정답, cur (N,) 현재값, chronos (N,H) 중앙값, gru (N,H, 있으면)."""
    horizons = list(range(1, H + 1))
    out = {}
    for sp in spaces:
        for m in pt.TARGETS:
            s = wide[(sp, m)]
            q = predict_fn(pt.contexts(s, points, ctx_len), H)
            d = {"y": pt.truth(s, points, horizons, 0).astype(float), "cur": s.reindex(points).to_numpy(dtype=float),
                 "chronos": q[:, :, 1].astype(float)}
            if gru and (sp, m) in gru:
                d["gru"] = gru[(sp, m)].astype(float)
            out[(sp, m)] = d
    return out


def fit_weights(col: dict[tuple[str, str], dict[str, np.ndarray]], step_min: int,
                report_minutes: list[int]) -> dict[tuple[str, int], float]:
    """(항목, 지평분) → GRU 가중치 w (예측 = w·GRU + (1-w)·Chronos). 공간을 합쳐 MAE 최소. GRU 없으면 빈 dict."""
    w = {}
    for m in pt.TARGETS:
        for hm in report_minutes:
            h = hm // step_min - 1
            ys, gs, cs = [], [], []
            for (_, mm), d in col.items():
                if mm != m or "gru" not in d:
                    continue
                ok = ~np.isnan(d["y"][:, h]) & ~np.isnan(d["gru"][:, h]) & ~np.isnan(d["chronos"][:, h])
                ys.append(d["y"][ok, h])
                gs.append(d["gru"][ok, h])
                cs.append(d["chronos"][ok, h])
            if not sum(len(y) for y in ys):
                continue
            y, g, c = (np.concatenate(a) for a in (ys, gs, cs))
            maes = [np.abs(a * g + (1 - a) * c - y).mean() for a in WEIGHT_GRID]
            w[(m, hm)] = float(WEIGHT_GRID[int(np.argmin(maes))])
    return w


def ensemble_rows(col: dict[tuple[str, str], dict[str, np.ndarray]], step_min: int, report_minutes: list[int],
                  weights: dict[tuple[str, int], float], extra: dict[str, dict] | None = None) -> pd.DataFrame:
    """같은 시점 MAE: 현재값 유지 / GRU / Chronos / 평균(0.5) / 가중평균(검증 w) [+ extra 방법들].

    extra: 방법 이름 → {(공간, 항목): (N,H) 예측}. 모든 방법이 값이 있는 시점만 채점."""
    rows = []
    for (sp, m), d in col.items():
        H = d["y"].shape[1]
        for hm in report_minutes:
            h = hm // step_min - 1
            preds = {"persistence": np.repeat(d["cur"][:, None], H, axis=1), "chronos": d["chronos"]}
            if "gru" in d:
                preds["gru"] = d["gru"]
                preds["mean_0.5"] = 0.5 * d["gru"] + 0.5 * d["chronos"]
                if (m, hm) in weights:
                    a = weights[(m, hm)]
                    preds["weighted"] = a * d["gru"] + (1 - a) * d["chronos"]
            for name, e in (extra or {}).items():
                if (sp, m) in e:
                    preds[name] = e[(sp, m)]
            ok = ~np.isnan(d["y"][:, h])
            for p in preds.values():
                ok &= ~np.isnan(p[:, h])
            if not ok.any():
                continue
            r = {"space": sp, "metric": m, "horizon_min": hm, "n": int(ok.sum())}
            for name, p in preds.items():
                r[name] = float(np.abs(p[ok, h] - d["y"][ok, h]).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def ensemble_wins(res: pd.DataFrame) -> pd.DataFrame:
    """앙상블이 GRU·Chronos 둘 다보다 MAE 가 작은 칸 수 (공간×항목×지평 칸 기준)."""
    rows = []
    if "gru" not in res:
        return pd.DataFrame()
    best = res[["gru", "chronos"]].min(axis=1)
    for name in ("mean_0.5", "weighted"):
        if name not in res:
            continue
        rows.append({"method": name, "cells": len(res),
                     "beats_both": int(((res[name] < res["gru"]) & (res[name] < res["chronos"])).sum()),
                     "beats_gru": int((res[name] < res["gru"]).sum()),
                     "beats_chronos": int((res[name] < res["chronos"]).sum()),
                     "mae_ratio_vs_better_single": round(float((res[name] / best).mean()), 3)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (a) Chronos-2 공변량

def try_load_chronos2(name: str, device: str, loader=None):
    """(파이프라인, 상태 문자열). 실패하면 (None, '설치 안 됨: …' / '불러오기 실패: …')."""
    if loader is None:
        try:
            import torch
            from chronos import Chronos2Pipeline
        except ImportError as e:
            return None, f"설치 안 됨 (Chronos2Pipeline import 실패: {e})"

        def loader(n, d):
            return Chronos2Pipeline.from_pretrained(n, device_map=d, torch_dtype=torch.float32)
    try:
        return loader(name, device), "ok"
    except ImportError as e:
        return None, f"설치 안 됨 ({e})"
    except Exception as e:   # 네트워크·캐시 없음·버전 불일치 등
        return None, f"불러오기 실패 ({type(e).__name__}: {str(e)[:120]})"


def _filled(s: pd.Series) -> np.ndarray:
    """공변량 결측은 앞값 → 뒷값 → 0 으로 채운다 (결측 처리는 목표 쪽만 믿는다)."""
    return s.ffill().bfill().fillna(0.0).to_numpy(dtype="float32")


def covariate_inputs(wide: pd.DataFrame, space: str, metric: str, points: pd.DatetimeIndex, ctx_len: int, H: int,
                     sched: dict[str, Any] | None = None, outdoor: str | None = None, tz: str = "Asia/Seoul",
                     step_min: int = 5, use_covariates: bool = True) -> list[dict[str, Any]]:
    """Chronos-2 입력 목록 (시점마다 dict 하나).

    공변량: 같은 공간의 나머지 두 항목·외기 온습도(과거만), 시간표 사용 여부(과거 + 앞으로 H칸, 미리 아는 값).
    창 길이는 ctx_len 으로 고정, 데이터 시작 전 칸은 목표 NaN·공변량 첫 값으로 채운다."""
    target = pt.contexts(wide[(space, metric)], points, ctx_len)
    if not use_covariates:
        return [{"target": t} for t in target]
    past = {f"own_{m}": _filled(wide[(space, m)]) for m in ("co2", "temperature", "humidity")
            if m != metric and (space, m) in wide}
    if outdoor:
        past.update({f"out_{m}": _filled(wide[(outdoor, m)]) for m in ("temperature", "humidity") if (outdoor, m) in wide})
    sched_arr = None
    if sched:
        from tsinfer.models.state import occupancy_labels

        full = wide.index.append(pd.date_range(wide.index[-1] + pd.Timedelta(minutes=step_min), periods=H,
                                               freq=f"{step_min}min"))
        sched_arr = occupancy_labels(full, [space], sched, tz=tz)[space].fillna(0.0).to_numpy(dtype="float32")
        past["schedule"] = sched_arr[:len(wide)]
    pos = wide.index.get_indexer(points)
    out = []
    for i, p in enumerate(pos):
        lo = p + 1 - ctx_len
        pad = max(0, -lo)
        pc = {k: np.r_[np.full(pad, a[0], "float32"), a[max(0, lo):p + 1]] for k, a in past.items()}
        item = {"target": target[i], "past_covariates": pc}
        if sched_arr is not None:
            item["future_covariates"] = {"schedule": sched_arr[p + 1:p + 1 + H]}
        out.append(item)
    return out


def chronos2_predict(pipe, inputs: list[dict[str, Any]], H: int, batch: int = 64) -> np.ndarray:
    """(N, H, 3) 분위수. Chronos-2 는 입력마다 (변수 수, H, 분위수) 텐서를 돌려준다."""
    outs = []
    for i in range(0, len(inputs), batch):
        q, _ = pipe.predict_quantiles(inputs[i:i + batch], prediction_length=H, quantile_levels=pt.QUANTILES)
        outs += [np.asarray(t.float().cpu().numpy())[0] for t in q]
    return np.stack(outs)


# ---------------------------------------------------------------- 실행

def run(args, cfg: dict[str, Any], wide: pd.DataFrame) -> list[str]:
    """보고서 줄 목록. args: pretrained.main 의 인자 (model, device, every, max_points, context_hours,
    chronos2_model, no_chronos2)."""
    import torch

    from tsinfer.data.split import split_by_time

    t_start = time.perf_counter()
    step = cfg["forecast"]["step_minutes"]
    H = cfg["forecast"]["horizon_minutes"] // step
    rep = cfg["forecast"]["report_horizons"]
    tz = cfg.get("timezone", "Asia/Seoul")
    _, valid, test = split_by_time(wide, cfg)
    spaces = [sp for sp in cfg["spaces"] if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in pt.TARGETS)]
    p_te = pt.eval_points(test.index, args.every, args.max_points, cfg["forecast"]["horizon_minutes"])
    p_va = pt.eval_points(valid.index, args.every, args.max_points, cfg["forecast"]["horizon_minutes"])
    ctx_len = args.context_hours * 60 // step

    pipe = pt.load_chronos(args.model, args.device)
    g_va, g_te = pt.gru_at(wide, p_va), pt.gru_at(wide, p_te)
    col_va = collect(wide, p_va, spaces, lambda c, h: pt.chronos_predict(pipe, c, h), ctx_len, H, g_va and g_va[0])
    col_te = collect(wide, p_te, spaces, lambda c, h: pt.chronos_predict(pipe, c, h), ctx_len, H, g_te and g_te[0])
    del pipe
    gc.collect()                      # Chronos-2 를 올리기 전에 Bolt 를 내린다 (8GB 공유 메모리)
    weights = fit_weights(col_va, step, rep)
    weight_note = ("검증 기간 시점에서 항목·지평별로 고름" if weights else
                   "GRU 가 없거나 검증 시점 GRU 예측이 없어 가중평균 행 없음 (평균은 0.5 고정)")

    c2_status, extra, c2_time = "건너뜀 (--no-chronos2)", {}, {}
    if not args.no_chronos2:
        pipe2, c2_status = try_load_chronos2(args.chronos2_model, args.device)
        if pipe2 is not None:
            sched = load_yaml("schedule.yaml")
            outdoor = cfg.get("model", {}).get("outdoor_space")
            outdoor = outdoor if outdoor in cfg["spaces"] else None
            try:
                for name, use_cov in (("chronos2", False), ("chronos2+cov", True)):
                    t0 = time.perf_counter()
                    extra[name] = {(sp, m): chronos2_predict(pipe2, covariate_inputs(
                        wide, sp, m, p_te, ctx_len, H, sched, outdoor, tz, step, use_cov), H)[:, :, 1]
                        for sp in spaces for m in pt.TARGETS}
                    c2_time[name] = round(time.perf_counter() - t0, 1)
            except Exception as e:   # 버전에 따라 입력 형식이 다를 수 있다
                c2_status = f"예측 실패 ({type(e).__name__}: {str(e)[:160]})"
                extra = {}
            del pipe2
            gc.collect()

    res = ensemble_rows(col_te, step, rep, weights, extra)
    methods = [c for c in ("persistence", "gru", "chronos", "mean_0.5", "weighted", "chronos2", "chronos2+cov")
               if c in res]
    avg = res.groupby(["metric", "horizon_min"])[methods].mean().round(3)
    w_tab = pd.Series({f"{m} {hm}분": w for (m, hm), w in weights.items()}, name="w_gru (GRU 몫)").to_frame()
    wins = ensemble_wins(res)
    c2_cols = [c for c in ("chronos", "chronos2", "chronos2+cov") if c in avg]
    return [
        "# 과제 6 심화. 앙상블 · 공변량 모델",
        "",
        "## 측정 조건",
        f"- 사전학습 모델: {args.model}, 장치 {args.device}, torch {torch.__version__}, CPU 스레드 {torch.get_num_threads()}, "
        "gpu-run.sh 락 안에서 실행",
        f"- 시험 시점: {test.index.min():%Y-%m-%d}~{test.index.max():%Y-%m-%d} 중 {args.every}분 간격 {len(p_te)}개 "
        f"(기본 보고서 06_pretrained.md 와 같은 고르는 법), 검증 시점(가중치 선택용) {len(p_va)}개",
        f"- GRU: models/forecast_gru.pt ({'있음' if g_te else '없음'})",
        "- 앙상블: 평균 = 0.5·GRU + 0.5·Chronos 중앙값 (고정), "
        f"가중평균 = w·GRU + (1-w)·Chronos, w ∈ 0~1 (0.05 간격) — {weight_note}",
        "- 주의: GRU 는 검증 기간으로 조기 종료했으므로 검증 기간 GRU 오차가 시험보다 약간 낙관적이다 → w 가 GRU 쪽으로 치우칠 수 있다",
        f"- Chronos-2 ({args.chronos2_model}): {c2_status}" + (f", 예측 시간(초) {c2_time}" if c2_time else ""),
        "- Chronos-2 공변량: 같은 공간의 나머지 두 항목·외기 온습도(과거만), 시간표 사용 여부(과거 + 앞으로 60분, 미리 아는 값). "
        "공변량 결측은 앞값으로 채움",
        "- 모든 방법이 값이 있는 시점만 채점 (같은 시점 비교)",
        f"- 실행 시간 {time.perf_counter() - t_start:.0f}s",
        "",
        "## 1. 공간 평균 MAE (CO₂ ppm / 온도 ℃)",
        "",
        avg.to_markdown(),
        "",
        "## 2. 앙상블이 둘 다보다 나은가 (공간×항목×지평 칸 수)",
        "",
        wins.to_markdown(index=False) if len(wins) else "GRU 가 없어 앙상블을 만들지 못함",
        "",
        "## 3. 검증 기간에서 고른 가중치",
        "",
        w_tab.to_markdown() if len(w_tab) else "(없음 — 가중평균 대신 0.5 고정 평균만)",
        "",
        "## 4. 공변량 모델 (Chronos-2, 공간 평균 MAE)",
        "",
        avg[c2_cols].to_markdown() if extra else f"| Chronos-2 |\n|:--|\n| {c2_status} |",
        "",
        "## 5. 공간별 MAE",
        "",
        res.set_index(["space", "metric", "horizon_min"])[["n", *methods]].round(3).to_markdown(),
        "",
        "## 해석 (직접 작성)",
        "- 앙상블이 GRU·Chronos 둘 다보다 나은 지평과 이유 (두 모델 오차 방향이 다른가):",
        "- 공변량(온습도·일정)이 Chronos-2 를 얼마나 바꿨나, 특히 수업 시작·끝 무렵:",
        "- 서비스에 넣을 가치 (모델 두 개를 돌리는 지연·메모리 대비 이득):",
        "",
    ]
