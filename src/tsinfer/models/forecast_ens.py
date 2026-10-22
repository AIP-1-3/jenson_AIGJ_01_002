"""과제 3 개선(C): GRU 여러 개의 평균 + Chronos-Bolt 섞기. 60분 CO₂ −30% 목표에 다시 도전한다.

    ~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.forecast_ens   # → reports/03c_forecast_ens.md
    옵션: --seeds 42 43 44 45 46  --lookbacks 60 120 240  --no-chronos  --from-cache

공정성 (시험 기간은 마지막에 보고만 한다)
- 구성원은 실행 전에 고정한다: GRU × 과거 길이(1·2·4시간) × 시드 5개 = 15개. 시험 성적으로 고르거나 빼지 않는다.
- 섞는 비율 w(GRU 평균 대 Chronos-Bolt)와 축소 계수 α 는 검증 기간(일하는 날)에서만 정한다.
- 후보 방식(단일 GRU / 평균·공통 α / 평균·공간별 α / 평균+Chronos)은 검증 기간을 날짜 홀짝으로 나눈
  교차 맞춤 점수로 고른다: 한쪽 날로 w·α 를 정하고 다른 쪽 날에서 채점한다. 계수가 많은 방식이 검증 기간에서만
  좋아 보이는 것(같은 날로 정하고 같은 날로 채점)을 막기 위해서다. 최종 w·α 는 검증 기간 전체로 다시 정한다.
- 평가 시점: 구성원 중 하나라도 창을 만들 수 있는 (시각, 공간). 현재값 유지도 같은 시점에서 계산한다.

설계
- 학습 부품은 forecast.py 그대로 (fit_spec / make_windows / train). 다른 것은 과거 길이와 시드뿐.
- 각 구성원의 '원래 단위 변화량'(α 없이)을 (시각, 공간) 열쇠로 모아 평균한다. 창이 없는 구성원은 그 시점 평균에서 빠진다.
- Chronos-Bolt 는 같은 열쇠에서 과거 42시간만 보고 예측한 중앙값. 패키지·모델을 못 불러오면 건너뛰고 사유를 남긴다.
- 구성원 예측은 models/forecast_ens_cache.pkl 에 저장한다 → --from-cache 로 학습 없이 섞기만 다시 할 수 있다.
"""
from __future__ import annotations

import argparse
import gc
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import forecast as fc
from tsinfer.models.forecast_deep import model_cfg

REPORT = ROOT / "reports" / "03c_forecast_ens.md"
CACHE = ROOT / "models" / "forecast_ens_cache.pkl"
LOOKBACKS = (60, 120, 240)
SEEDS = (42, 43, 44, 45, 46)
ALPHA_GRID = np.round(np.arange(0, 1.21, 0.1), 2)
W_GRID = np.round(np.arange(0, 1.0001, 0.1), 2)
VARIANTS = ("single", "ens", "ens_space", "ens_chronos")
VARIANT_NAMES = {"single": "단일 GRU 2시간(시드 42), 공통 α", "ens": "GRU 평균, 공통 α",
                 "ens_space": "GRU 평균, 공간별 α", "ens_chronos": "GRU 평균 + Chronos-Bolt, 공간별 w·α"}


# ---------------------------------------------------------------- 구성원 학습

def member_deltas(model, spec: fc.FeatureSpec, x: np.ndarray, info: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """α 없이 원래 단위 변화량. 항목 → (시각, 공간) 열쇠 DataFrame (열 = 지평 칸 1..H)."""
    raw = fc.decode(fc.predict_array(model, x), info, fc.FeatureSpec(**{**spec.to_dict(), "alpha": {}}))
    key = pd.MultiIndex.from_arrays([pd.DatetimeIndex(info["time"]), info["space"].to_numpy()], names=["time", "space"])
    return {m: pd.DataFrame(raw[m] - info[f"cur_{m}"].to_numpy()[:, None], index=key, columns=spec.horizons)
            for m in fc.TARGETS}


def train_members(cfg: dict[str, Any], wide: pd.DataFrame, schedule, lookbacks, seeds,
                  deadline: float | None = None, log=print) -> list[dict[str, Any]]:
    from tsinfer.data.split import split_by_time

    train_df, valid_df, test_df = split_by_time(wide, cfg)
    members = []
    for lb in lookbacks:
        for seed in seeds:
            if deadline is not None and time.time() > deadline:
                log(f"시간 예산 초과 → GRU {lb}분 시드 {seed} 건너뜀")
                continue
            c = model_cfg(cfg, "gru", lb, cfg["model"]["epochs"], cfg["model"]["patience"])
            c["model"]["seed"] = seed
            spec = fc.fit_spec(wide, train_df.index, c, schedule=schedule)
            va_index = valid_df.index[fc.is_workday(valid_df.index, spec)]
            if len(va_index) < len(valid_df) * 0.2:
                va_index = valid_df.index
            x_tr, y_tr, _ = fc.make_windows(wide, spec, train_df.index)
            x_va, y_va, info_va = fc.make_windows(wide, spec, va_index)
            x_te, _, info_te = fc.make_windows(wide, spec, test_df.index, with_target=False)
            log(f"GRU {lb}분 시드 {seed}: 창 train {len(x_tr):,} / valid {len(x_va):,} / test {len(x_te):,}")
            t0 = time.perf_counter()
            model, history = fc.train(x_tr, y_tr, x_va, y_va, spec, c["model"], log=lambda s: log("  " + s))
            sec = time.perf_counter() - t0
            members.append({"lookback": lb, "seed": seed, "train_sec": sec, "epochs": len(history),
                            "best_valid_loss": min(h["valid"] for h in history) if history else float("nan"),
                            "valid": member_deltas(model, spec, x_va, info_va),
                            "test": member_deltas(model, spec, x_te, info_te)})
            log(f"  → {len(history)}에폭 {sec:.0f}s")
            del model, x_tr, y_tr, x_va, y_va, x_te
            gc.collect()
    return members


# ---------------------------------------------------------------- 모으기

def gather(members: list[dict[str, Any]], part: str, wide: pd.DataFrame, step_min: int) -> dict[str, Any]:
    """구성원 변화량을 열쇠 합집합에 맞춰 쌓는다. 반환: keys, 항목별 cur (K,), truth (K,H), stack (M,K,H)."""
    keys = members[0][part]["co2"].index
    for mem in members[1:]:
        keys = keys.union(mem[part]["co2"].index)
    keys = keys.sort_values()
    times = pd.DatetimeIndex(keys.get_level_values("time"))
    spaces = keys.get_level_values("space").to_numpy()
    H = members[0][part]["co2"].shape[1]
    out = {"keys": keys, "time": times, "space": spaces, "cur": {}, "truth": {}, "stack": {}}
    for m in fc.TARGETS:
        cur = np.full(len(keys), np.nan)
        truth = np.full((len(keys), H), np.nan)
        for sp in np.unique(spaces):
            sel = spaces == sp
            s = wide[(sp, m)]
            cur[sel] = s.reindex(times[sel]).to_numpy(dtype=float)
            for k in range(H):
                truth[sel, k] = s.reindex(times[sel] + pd.Timedelta(minutes=(k + 1) * step_min)).to_numpy(dtype=float)
        out["cur"][m], out["truth"][m] = cur, truth
        out["stack"][m] = np.stack([mem[part][m].reindex(keys).to_numpy(dtype=float) for mem in members])
    return out


def chronos_deltas(g: dict[str, Any], wide: pd.DataFrame, predict_fn, ctx_len: int, H: int) -> dict[str, np.ndarray]:
    """같은 열쇠에서 Chronos 중앙값 − 현재값 (K,H)."""
    from tsinfer.models import pretrained as pt

    out = {}
    for m in fc.TARGETS:
        d = np.full((len(g["keys"]), H), np.nan)
        for sp in np.unique(g["space"]):
            sel = g["space"] == sp
            q = predict_fn(pt.contexts(wide[(sp, m)], g["time"][sel], ctx_len), H)
            d[sel] = q[:, :, 1] - g["cur"][m][sel, None]
        out[m] = d
    return out


# ---------------------------------------------------------------- 섞는 비율·축소 계수

def fit_cell(dg: np.ndarray, dc: np.ndarray | None, cur: np.ndarray, truth: np.ndarray,
             use_w: bool) -> tuple[float, float]:
    """한 지평에서 MAE 가 가장 작은 (w, α). 예측 = 현재값 + α·((1−w)·GRU 변화량 + w·Chronos 변화량)."""
    ok = ~np.isnan(truth) & ~np.isnan(dg) & ~np.isnan(cur)
    if use_w and dc is not None:
        ok &= ~np.isnan(dc)
    if not ok.any():
        return 0.0, 1.0
    best = (np.inf, 0.0, 1.0)
    for w in (W_GRID if use_w and dc is not None else (0.0,)):
        mix = dg[ok] if w == 0 else (1 - w) * dg[ok] + w * dc[ok]
        err = np.abs(cur[ok, None] + mix[:, None] * ALPHA_GRID[None, :] - truth[ok, None]).mean(axis=0)
        i = int(np.argmin(err))
        if err[i] < best[0] - 1e-12:
            best = (err[i], float(w), float(ALPHA_GRID[i]))
    return best[1], best[2]


def gru_delta(g: dict[str, Any], m: str, variant: str, single_idx: int | None) -> np.ndarray:
    if variant == "single":
        return g["stack"][m][single_idx]
    with np.errstate(all="ignore"):
        return np.nanmean(g["stack"][m], axis=0)


def fit_params(g: dict[str, Any], chron: dict[str, np.ndarray] | None, variant: str, single_idx: int | None,
               sel: np.ndarray | None = None) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """(공간, 항목) → 지평별 (w, α). sel: 맞출 때 쓸 열쇠 (None = 전부)."""
    sel = np.ones(len(g["keys"]), bool) if sel is None else sel
    per_space = variant in ("ens_space", "ens_chronos")
    use_w = variant == "ens_chronos"
    params = {}
    for m in fc.TARGETS:
        dg = gru_delta(g, m, variant, single_idx)
        dc = chron[m] if (use_w and chron is not None) else None
        H = dg.shape[1]
        groups = [(sp, sel & (g["space"] == sp)) for sp in np.unique(g["space"])] if per_space else [(None, sel)]
        for sp, s in groups:
            cells = [fit_cell(dg[s, k], None if dc is None else dc[s, k], g["cur"][m][s], g["truth"][m][s, k], use_w)
                     for k in range(H)]
            for sp2 in ([sp] if sp is not None else np.unique(g["space"])):
                params[(sp2, m)] = cells
    return params


def apply_params(g: dict[str, Any], chron: dict[str, np.ndarray] | None, variant: str, single_idx: int | None,
                 params: dict[tuple[str, str], list[tuple[float, float]]]) -> dict[str, np.ndarray]:
    out = {}
    for m in fc.TARGETS:
        dg = gru_delta(g, m, variant, single_idx)
        pred = np.full(dg.shape, np.nan)
        for sp in np.unique(g["space"]):
            s = g["space"] == sp
            wa = np.array(params[(sp, m)])                        # (H, 2)
            mix = dg[s]
            if variant == "ens_chronos" and chron is not None:
                dc = chron[m][s]
                mix = np.where(np.isnan(dc), dg[s], (1 - wa[None, :, 0]) * dg[s] + wa[None, :, 0] * dc)
            pred[s] = g["cur"][m][s, None] + wa[None, :, 1] * mix
        out[m] = pred
    return out


def mae_rows(g: dict[str, Any], pred: dict[str, np.ndarray], method: str, report_minutes: list[int],
             step_min: int) -> pd.DataFrame:
    """forecast.score 와 같은 모양: 공간·항목·지평별 MAE 와 같은 시점의 현재값 유지 MAE."""
    rows = []
    for m in fc.TARGETS:
        for hm in report_minutes:
            k = hm // step_min - 1
            truth, p, cur = g["truth"][m][:, k], pred[m][:, k], g["cur"][m]
            ok = ~np.isnan(truth) & ~np.isnan(p) & ~np.isnan(cur)
            for sp in sorted(np.unique(g["space"])):
                s = ok & (g["space"] == sp)
                if s.any():
                    rows.append({"space": sp, "metric": m, "horizon_min": hm, "n": int(s.sum()),
                                 method: float(np.abs(p[s] - truth[s]).mean()),
                                 "persistence": float(np.abs(cur[s] - truth[s]).mean())})
    return pd.DataFrame(rows)


def ratio_score(rows: pd.DataFrame, method: str) -> float:
    """선택 기준 (낮을수록 좋음): 공간별 60분 CO₂ MAE ÷ 현재값 유지 의 평균. forecast_deep.val_score 와 같다."""
    r = rows[(rows["metric"] == "co2") & (rows["horizon_min"] == 60)]
    return float((r[method] / r["persistence"]).mean()) if len(r) else float("inf")


def crossfit_score(g: dict[str, Any], chron, variant: str, single_idx: int | None, tz: str, step_min: int) -> float:
    """검증 기간을 날짜 홀짝으로 나눠 한쪽으로 맞추고 다른 쪽에서 채점 → 두 쪽 예측을 합쳐 점수."""
    day = pd.DatetimeIndex(g["time"]).tz_convert(tz).normalize()
    fold = (day.dayofyear.to_numpy() % 2).astype(int)
    pred = {m: np.full(g["truth"][m].shape, np.nan) for m in fc.TARGETS}
    for f in (0, 1):
        params = fit_params(g, chron, variant, single_idx, sel=fold != f)
        p = apply_params(g, chron, variant, single_idx, params)
        for m in fc.TARGETS:
            pred[m][fold == f] = p[m][fold == f]
    return ratio_score(mae_rows(g, pred, variant, [60], step_min), variant)


# ---------------------------------------------------------------- 보고서

def load_chronos_fn(model: str, device: str):
    """(predict_fn, 상태 문구). 못 불러오면 (None, 사유)."""
    try:
        from tsinfer.models import pretrained as pt

        pipe = pt.load_chronos(model, device)
        return (lambda c, h: pt.chronos_predict(pipe, c, h)), f"{model} ({device})"
    except Exception as e:   # 패키지 없음·다운로드 실패 등 — 앙상블은 GRU 만으로 계속
        return None, f"건너뜀 ({type(e).__name__}: {str(e)[:120]})"


def check_table(res: pd.DataFrame, method: str) -> pd.DataFrame:
    t = fc.target_check(res, method)
    return t.rename(columns={"space": "공간", "persistence": "유지 60분", method: "모델 60분", "improve_%": "개선 %",
                             "pass(≥30%)": "−30% 통과", "temp_all_horizons_better": "온도 전 지평 우세"})


def write_report(cfg, args, test_df, members, g_te, scores, chosen, results, chron_status, params,
                 total_sec) -> str:
    mc = cfg["model"]
    lines = [
        "# 과제 3 개선(C). GRU 평균 + Chronos-Bolt 섞기",
        "",
        "## 측정 조건",
        f"- 학습 ~{cfg['split']['train_end']}, 검증 ~{cfg['split']['valid_end']}(일하는 날만), 시험 "
        f"{test_df.index.min():%Y-%m-%d} ~ {test_df.index.max():%Y-%m-%d} (UTC)",
        f"- 구성원(실행 전 고정): GRU × 과거 {', '.join(f'{lb}분' for lb in args.lookbacks)} × 시드 "
        f"{', '.join(map(str, args.seeds))} = {len(members)}개 학습. hidden {mc['hidden']}, 에폭 상한 {mc['epochs']}, "
        f"patience {mc['patience']}, 일정 입력 {'없음' if args.no_schedule else '있음'}",
        f"- Chronos-Bolt: {chron_status}, 과거 {args.context_hours}시간",
        "- w(GRU 평균 대 Chronos 비율)·α(변화량 축소)는 검증 기간에서만 정함. 방식 선택은 검증 기간 날짜 홀짝 교차 맞춤 점수.",
        f"- 평가 시점: 시험 기간에서 구성원 중 하나라도 창을 만들 수 있는 (시각, 공간) {len(g_te['keys']):,}개, "
        "현재값 유지도 같은 시점",
        f"- 장비: Jetson Orin Nano Super, PyTorch CPU {mc.get('threads', 3)}스레드, 전체 {total_sec / 60:.0f}분",
        "",
        "## 1. 방식 선택 (검증 기간 교차 맞춤 점수, 낮을수록 좋음)",
        "",
        pd.DataFrame([{"방식": VARIANT_NAMES[v], "검증 점수(60분 CO₂ ÷ 유지)": round(s, 3),
                       "선택": "✔" if v == chosen else ""} for v, s in scores.items()]).to_markdown(index=False),
        "",
        f"**선택: {VARIANT_NAMES[chosen]}** — 시험 결과를 보기 전에 정해짐.",
        "",
        "## 2. 과제 3 기준 확인 (시험 기간, 선택된 방식)",
        "",
        check_table(results[chosen], chosen).to_markdown(index=False),
        "",
        "## 3. 방식별 시험 기간 60분 CO₂ 개선율 % (보고용 — 선택에는 쓰지 않음)",
        "",
    ]
    rows = []
    for v, res in results.items():
        r = res[(res["metric"] == "co2") & (res["horizon_min"] == 60)]
        row = {"방식": VARIANT_NAMES[v]}
        for _, x in r.iterrows():
            row[x["space"]] = round((1 - x[v] / x["persistence"]) * 100, 1)
        rows.append(row)
    lines += [pd.DataFrame(rows).to_markdown(index=False), "",
              "## 4. 선택된 방식의 시험 기간 MAE (CO₂ ppm / 온도 ℃)", "",
              fc.report_table(results[chosen], [chosen]).to_markdown(), "",
              "## 5. 구성원", ""]
    lines += [pd.DataFrame([{"과거(분)": mem["lookback"], "시드": mem["seed"], "에폭": mem["epochs"],
                             "최저 검증 손실": round(mem["best_valid_loss"], 4), "학습(s)": round(mem["train_sec"])}
                            for mem in members]).to_markdown(index=False), ""]
    p60 = [{"공간": sp, "항목": m, "w(60분)": wa[-1][0], "α(60분)": wa[-1][1]} for (sp, m), wa in sorted(params.items())]
    lines += ["## 6. 선택된 방식의 60분 계수 (검증 기간 전체로 결정)", "", pd.DataFrame(p60).to_markdown(index=False), "",
              "## 해석 (직접 작성)",
              "- 평균을 내니 단일 GRU 보다 무엇이 나아졌나 (공간별):",
              "- Chronos 를 섞는 비율이 공간마다 다른 이유:",
              "- −30% 를 넘긴/못 넘긴 공간과 그 이유:",
              "- 서비스에 넣는다면 비용(모델 15개·Chronos)과 이득을 어떻게 저울질할지:", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lookbacks", type=int, nargs="+", default=list(LOOKBACKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--no-chronos", action="store_true")
    ap.add_argument("--chronos-model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--context-hours", type=int, default=42)
    ap.add_argument("--from-cache", action="store_true", help="학습 없이 저장된 구성원 예측으로 섞기만")
    ap.add_argument("--no-schedule", action="store_true")
    ap.add_argument("--budget-min", type=float, default=240, help="넘기면 남은 구성원 학습을 건너뜀")
    args = ap.parse_args()

    from tsinfer.data.preprocess import clean, load_raw, to_wide
    from tsinfer.data.split import split_by_time

    t_start = time.time()
    cfg = load_yaml("default.yaml")
    step = cfg["forecast"]["step_minutes"]
    H = cfg["forecast"]["horizon_minutes"] // step
    rep = cfg["forecast"]["report_horizons"]
    tz = cfg.get("timezone", "Asia/Seoul")
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    test_df = split_by_time(wide, cfg)[2]
    schedule = None if args.no_schedule else load_yaml("schedule.yaml")

    if args.from_cache:
        members = pickle.loads(CACHE.read_bytes())
        print(f"저장된 구성원 {len(members)}개 불러옴: {CACHE}")
    else:
        members = train_members(cfg, wide, schedule, args.lookbacks, args.seeds,
                                t_start + args.budget_min * 60)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_bytes(pickle.dumps(members))
        print(f"구성원 예측 저장: {CACHE}")
    if not members:
        print("학습된 구성원이 없습니다")
        return 1
    single_idx = next((i for i, mem in enumerate(members) if mem["lookback"] == 120 and mem["seed"] == 42), 0)

    g_va, g_te = gather(members, "valid", wide, step), gather(members, "test", wide, step)
    chron_va = chron_te = None
    chron_status = "건너뜀 (--no-chronos)"
    if not args.no_chronos:
        fn, chron_status = load_chronos_fn(args.chronos_model, args.device)
        if fn is not None:
            t0 = time.perf_counter()
            ctx_len = args.context_hours * 60 // step
            chron_va = chronos_deltas(g_va, wide, fn, ctx_len, H)
            chron_te = chronos_deltas(g_te, wide, fn, ctx_len, H)
            chron_status += f", 예측 {time.perf_counter() - t0:.0f}s"
            print(f"Chronos 완료: {chron_status}")

    variants = [v for v in VARIANTS if v != "ens_chronos" or chron_va is not None]
    scores = {v: crossfit_score(g_va, chron_va, v, single_idx, tz, step) for v in variants}
    chosen = min(scores, key=scores.get)
    print("검증 교차 맞춤 점수:", {v: round(s, 4) for v, s in scores.items()}, "→ 선택", chosen)

    results, final_params = {}, {}
    for v in variants:
        params = fit_params(g_va, chron_va, v, single_idx)
        results[v] = mae_rows(g_te, apply_params(g_te, chron_te, v, single_idx, params), v, rep, step)
        final_params[v] = params
    text = write_report(cfg, args, test_df, members, g_te, scores, chosen, results, chron_status, final_params[chosen],
                        time.time() - t_start)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {REPORT} (전체 {(time.time() - t_start) / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
