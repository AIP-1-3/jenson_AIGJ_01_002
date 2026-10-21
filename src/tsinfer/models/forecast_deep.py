"""과제 3 심화(B): 모델 3종(GRU·LSTM·TCN) × 과거 길이 비교, 처음 보는 공간 일반화.

    python -m tsinfer.models.forecast_deep                # (A) 5개 학습 + (B) 한 공간 빼기 → reports/03b_forecast_deep.md
    python -m tsinfer.models.forecast_deep --full-grid    # (A) 3종 × 1·2·4시간 9개 전부 (약 2배 시간)
    python -m tsinfer.models.forecast_deep --only arch    # (A)만 / --only loso 는 (B)만

설계
- 학습·평가 부품은 forecast.py 것을 그대로 쓴다 (fit_spec / make_windows / train / fit_alpha / decode / score).
  그래서 입력 특성·변화량 목표·지평별 α·일하는 날 검증이 과제 3 본 모델과 같다. 다른 것은 구조와 과거 길이뿐.
- (A) 시간 예산 때문에 9조합 대신 2단계: ① 3종 @ 2시간 → ② 검증 오차가 가장 낮은 구조 @ 1·4시간 = 5개 학습.
  최적 조합은 검증 기간 점수로만 고른다 (시험 기간은 보고만). 검증 점수 = 공간별 '60분 CO₂ MAE ÷ 현재값 유지' 평균
  — 공간마다 오차 크기가 달라 비율로 맞춰 평균한다. α 도 같은 검증 기간에서 정하므로 모든 조합에 똑같이 조금 낙관적이다.
- (B) 한 공간 빼기: 그 공간의 데이터는 정규화 통계·학습·검증·α 결정 어디에도 쓰지 않는다.
  그 공간의 정규화 통계는 나머지 공간 통계의 평균으로 대신하고, 공간 표시(one-hot)는 모두 0 으로 둔다.
  비교 대상(같은 공간 포함 학습)도 같은 에폭 상한으로 다시 학습해 조건을 맞춘다.
- 일정 입력(configs/schedule.yaml)은 본 모델(run_all 의 --schedule)과 같게 기본으로 넣는다. --no-schedule 로 끈다.
- 심화 모델은 저장하지 않는다 (models/forecast_gru.pt 는 그대로).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import time
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import forecast as fc

REPORT = ROOT / "reports" / "03b_forecast_deep.md"
LOOKBACKS = (60, 120, 240)          # 분
STAGE1_LOOKBACK = 120


def model_cfg(cfg: dict[str, Any], arch: str, lookback_min: int, epochs: int, patience: int) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["model"].update(arch=arch, lookback_minutes=lookback_min, epochs=epochs, patience=patience)
    return c


def pooled_stats(spec: fc.FeatureSpec, space: str) -> None:
    """학습에 없던 공간의 정규화 통계를 학습 공간 통계의 평균으로 채운다 (그 공간 데이터는 보지 않는다)."""
    for m in fc.OWN:
        keys = [f"{sp}|{m}" for sp in spec.spaces if f"{sp}|{m}" in spec.mean]
        if keys:
            spec.mean[f"{space}|{m}"] = float(np.mean([spec.mean[k] for k in keys]))
            spec.std[f"{space}|{m}"] = float(np.mean([spec.std[k] for k in keys]))
    for m in fc.TARGETS:
        spec.target_std[f"{space}|{m}"] = float(np.mean([spec.target_std[f"{sp}|{m}"] for sp in spec.spaces]))


def fit_eval(cfg: dict[str, Any], wide: pd.DataFrame, schedule: dict[str, Any] | None = None,
             holdout: str | None = None, log=print) -> dict[str, Any]:
    """한 조합을 학습하고 검증·시험 점수를 낸다. holdout 이 있으면 그 공간을 완전히 빼고 학습해 그 공간만 시험."""
    from tsinfer.data.split import split_by_time

    train_df, valid_df, test_df = split_by_time(wide, cfg)
    if holdout is not None:
        # 빼는 공간 열을 지운 표로 통계를 계산 → 평균·표준편차·변화량 표준편차에 그 공간이 섞이지 않는다
        seen = wide.drop(columns=holdout, level="space")
        spec = fc.fit_spec(seen, train_df.index, cfg, schedule=schedule)
        if holdout in spec.spaces:
            raise ValueError(f"{holdout} 이 학습 공간에 남아 있습니다")
    else:
        spec = fc.fit_spec(wide, train_df.index, cfg, schedule=schedule)
    va_index = valid_df.index[fc.is_workday(valid_df.index, spec)]
    if len(va_index) < len(valid_df) * 0.2:
        va_index = valid_df.index
    x_tr, y_tr, info_tr = fc.make_windows(wide, spec, train_df.index)
    x_va, y_va, info_va = fc.make_windows(wide, spec, va_index)
    log(f"  창 train {len(x_tr):,} / valid {len(x_va):,}, 입력 {spec.lookback}칸×{spec.n_features}, "
        f"학습 공간 {spec.spaces}")
    t0 = time.perf_counter()
    model, history = fc.train(x_tr, y_tr, x_va, y_va, spec, cfg["model"], log=lambda s: log("  " + s))
    train_sec = time.perf_counter() - t0
    pred_va = fc.predict_array(model, x_va)
    spec.alpha = fc.fit_alpha(pred_va, info_va, wide, spec)
    rep = cfg["forecast"]["report_horizons"]
    val = fc.score(fc.decode(pred_va, info_va, spec), info_va, wide, spec, rep, "model")

    if holdout is not None:
        pooled_stats(spec, holdout)
        x_te, _, info_te = fc.make_windows(wide, spec, test_df.index, with_target=False, spaces=[holdout])
        spec_te = dataclasses.replace(spec, spaces=[holdout])     # score 는 spec.spaces 로 공간을 고른다
    else:
        x_te, _, info_te = fc.make_windows(wide, spec, test_df.index, with_target=False)
        spec_te = spec
    t1 = time.perf_counter()
    pred_te = fc.decode(fc.predict_array(model, x_te), info_te, spec)
    infer_ms = (time.perf_counter() - t1) / max(len(x_te), 1) * 1000
    test = fc.score(pred_te, info_te, wide, spec_te, rep, "model")
    return {"spec": spec, "val": val, "test": test, "history": history, "train_sec": train_sec,
            "params": fc.n_params(model), "infer_ms": infer_ms,
            "train_spaces": sorted(set(info_tr["space"])) if len(info_tr) else []}


def val_score(val: pd.DataFrame) -> float:
    """선택 기준: 공간별 60분 CO₂ MAE ÷ 현재값 유지 의 평균 (낮을수록 좋음)."""
    r = val[(val["metric"] == "co2") & (val["horizon_min"] == 60)]
    return float((r["model"] / r["persistence"]).mean()) if len(r) else float("inf")


# ---------------------------------------------------------------- (A) 구조 × 과거 길이

def run_arch(cfg: dict[str, Any], wide: pd.DataFrame, schedule=None, full_grid: bool = False,
             epochs: int = 20, patience: int = 5, deadline: float | None = None, log=print):
    """반환: 조합별 결과 dict 목록, 2단계에 쓴 구조, 건너뛴 조합."""
    results, skipped = [], []

    def one(arch: str, lb: int):
        if deadline is not None and time.time() > deadline:
            log(f"[A] 시간 예산 초과 → {arch}@{lb}분 건너뜀")
            skipped.append(f"{arch}@{lb}분")
            return
        log(f"[A] {arch.upper()} 과거 {lb}분 학습 시작")
        r = fit_eval(model_cfg(cfg, arch, lb, epochs, patience), wide, schedule, log=log)
        r.update(arch=arch, lookback_min=lb, score=val_score(r["val"]))
        log(f"[A] {arch.upper()}@{lb}분: 검증 점수 {r['score']:.3f}, {len(r['history'])}에폭 {r['train_sec']:.0f}s")
        results.append(r)

    if full_grid:
        for lb in LOOKBACKS:
            for arch in fc.ARCHS:
                one(arch, lb)
        stage2 = None
    else:
        for arch in fc.ARCHS:
            one(arch, STAGE1_LOOKBACK)
        done = [r for r in results if r["lookback_min"] == STAGE1_LOOKBACK]
        stage2 = min(done, key=lambda r: r["score"])["arch"] if done else "gru"
        log(f"[A] 2시간에서 검증 점수가 가장 낮은 구조: {stage2.upper()} → 1·4시간 학습")
        for lb in LOOKBACKS:
            if lb != STAGE1_LOOKBACK:
                one(stage2, lb)
    return results, stage2, skipped


def arch_tables(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, rows = [], []
    for r in results:
        label = f"{r['arch'].upper()} {r['lookback_min']}분"
        h = r["history"]
        best_ep = int(np.argmin([x["valid"] for x in h])) + 1 if h else 0
        summary.append({"조합": label, "검증 점수(60분 CO₂ ÷ 유지)": round(r["score"], 3),
                        "파라미터": r["params"], "에폭(최적/실행)": f"{best_ep}/{len(h)}",
                        "학습 시간(s)": round(r["train_sec"]), "에폭당(s)": round(r["train_sec"] / max(len(h), 1), 1),
                        "추론(ms/창)": round(r["infer_ms"], 3)})
        t = r["test"]
        for sp in t["space"].unique():
            g = t[t["space"] == sp].set_index(["metric", "horizon_min"])
            row = {"조합": label, "공간": sp}
            for hm in (5, 15, 30, 60):
                if ("co2", hm) in g.index:
                    row[f"CO₂ {hm}분"] = g.loc[("co2", hm), "model"]
            if ("co2", 60) in g.index:
                p = g.loc[("co2", 60), "persistence"]
                row["CO₂ 60분 유지"] = p
                row["60분 개선 %"] = (1 - g.loc[("co2", 60), "model"] / p) * 100
            if ("temperature", 60) in g.index:
                row["온도 60분"] = g.loc[("temperature", 60), "model"]
                row["온도 60분 유지"] = g.loc[("temperature", 60), "persistence"]
            rows.append(row)
    s = pd.DataFrame(summary)
    if len(s):
        s = s.sort_values("검증 점수(60분 CO₂ ÷ 유지)").reset_index(drop=True)
    return s, pd.DataFrame(rows).round(2)


# ---------------------------------------------------------------- (B) 한 공간 빼기

def run_loso(cfg: dict[str, Any], wide: pd.DataFrame, schedule=None, arch: str = "gru", lookback: int = 120,
             epochs: int = 12, patience: int = 4, deadline: float | None = None, log=print):
    c = model_cfg(cfg, arch, lookback, epochs, patience)
    spaces = fc.model_spaces(wide, cfg)
    log(f"[B] 기준: 모든 공간 포함 {arch.upper()}@{lookback}분 (같은 에폭 상한 {epochs})")
    ref = fit_eval(c, wide, schedule, log=log)
    rows, skipped, info = [], [], {}
    for sp in spaces:
        if deadline is not None and time.time() > deadline:
            log(f"[B] 시간 예산 초과 → {sp} 빼기 건너뜀")
            skipped.append(sp)
            continue
        log(f"[B] {sp} 를 학습에서 빼고 학습")
        r = fit_eval(c, wide, schedule, holdout=sp, log=log)
        info[sp] = r
        for m in fc.TARGETS:
            for hm in cfg["forecast"]["report_horizons"]:
                a = r["test"].query("metric == @m and horizon_min == @hm")
                b = ref["test"].query("space == @sp and metric == @m and horizon_min == @hm")
                if a.empty:
                    continue
                rows.append({"빼고 시험한 공간": sp, "항목": m, "지평(분)": hm, "n": int(a["n"].iloc[0]),
                             "처음 보는 공간(빼고 학습)": a["model"].iloc[0],
                             "포함 학습": b["model"].iloc[0] if len(b) else np.nan,
                             "현재값 유지": a["persistence"].iloc[0]})
    t = pd.DataFrame(rows)
    if len(t):
        t["빼고 학습 개선 %"] = (1 - t["처음 보는 공간(빼고 학습)"] / t["현재값 유지"]) * 100
        t["포함 학습 개선 %"] = (1 - t["포함 학습"] / t["현재값 유지"]) * 100
    return t.round(2), ref, info, skipped


# ---------------------------------------------------------------- 보고서

def schedule_link() -> list[str]:
    """회의실 '미래에 알 수 있는 정보' 항목은 과제 3 본 실행(--schedule)의 일정 포함/제외 결과를 재사용한다."""
    lines = ["## 미래에 알 수 있는 정보(일정) 입력 — 과제 3 본 실행 결과 재사용", "",
             "- 설계: 지금·15·30·60분 뒤 그 공간이 시간표상 사용 중인지(0/1, 공휴일·주말은 0)를 매 칸 입력에 붙인다 "
             "(forecast.SCHEDULE_OFFSETS). 미래 시각의 일정은 예측 시점에 이미 알 수 있으므로 미래 누설이 아니다.",
             "- 회의실 예약표를 받으면 configs/schedule.yaml 의 회의실 항목만 바꾸면 같은 입력으로 들어간다 "
             "(지금은 예약표가 없어 평일 9~18시 가정).", ""]
    csv = ROOT / "reports" / "03_forecast_mae.csv"
    if csv.exists():
        r = pd.read_csv(csv)
        if "gru_no_schedule" in r:
            r = r[(r["metric"] == "co2")][["space", "horizon_min", "persistence", "gru", "gru_no_schedule"]].copy()
            r["일정 효과(ppm)"] = r["gru_no_schedule"] - r["gru"]
            lines += ["reports/03_forecast_mae.csv 에서 CO₂ 부분 (gru = 일정 포함, gru_no_schedule = 일정 제외):", "",
                      r.sort_values(["space", "horizon_min"]).round(2).to_markdown(index=False), ""]
        else:
            lines += ["(03_forecast_mae.csv 에 일정 제외 열이 없음 — `python -m tsinfer.models.forecast --schedule` 로 다시 만든다)", ""]
    else:
        lines += ["(reports/03_forecast_mae.csv 없음 — `python -m tsinfer.models.forecast --compare-time --schedule` 먼저 실행)", ""]
    return lines


def write_report(cfg, wide, schedule, args, arch_res, stage2, skipped_a, loso, total_sec) -> str:
    from tsinfer.data.split import split_by_time

    _, _, test_df = split_by_time(wide, cfg)
    mc = cfg["model"]
    design = ("3종 × 1·2·4시간 전체 격자" if args.full_grid
              else f"3종 @ 2시간 → 검증 최고 구조({(stage2 or '-').upper()}) @ 1·4시간")
    lines = [
        "# 과제 3 심화. 모델 3종 × 과거 길이 비교, 처음 보는 공간 일반화",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터, 학습 ~{cfg['split']['train_end']}, 검증 ~{cfg['split']['valid_end']}(일하는 날만), "
        f"시험 {test_df.index.min():%Y-%m-%d} ~ {test_df.index.max():%Y-%m-%d} (UTC)",
        "- 입력·출력·목표는 과제 3 본 모델과 같음 (변화량 12칸 한 번에, 지평별 α 검증 결정, 공휴일 제외), "
        f"일정 입력 {'있음 (source=' + str(schedule.get('source')) + ')' if schedule else '없음'}",
        f"- 공통 하이퍼파라미터: hidden {mc['hidden']}, layers {mc['layers']}, dropout {mc['dropout']}, Adam lr {mc['lr']}, "
        f"batch {mc['batch_size']}, L1 손실, seed {mc['seed']}, TCN 채널 {mc.get('tcn_channels', 48)}·커널 3·dilation 1,2,4,8,16",
        f"- (A) 에폭 상한 {args.epochs}, patience {args.patience} / (B) 에폭 상한 {args.loso_epochs}, patience {args.loso_patience} "
        "(시간 예산 때문에 과제 3 본 실행의 40/6 보다 낮춤)",
        f"- (A) 설계: {design}. "
        "최적 조합은 검증 점수(공간별 60분 CO₂ MAE ÷ 현재값 유지 의 평균)로만 고르고, 시험 오차는 보고만 한다.",
        f"- 장비: Jetson Orin Nano Super, PyTorch CPU {mc.get('threads', 3)}스레드, 전체 실행 {total_sec / 60:.0f}분",
        "",
    ]
    if arch_res is not None:
        summ, tab = arch_tables(arch_res)
        lines += ["## (A) 조합별 요약 (검증 점수 낮은 순 — 맨 위가 선택된 최적 조합)", "",
                  summ.to_markdown(index=False) if len(summ) else "(학습된 조합 없음)", ""]
        if skipped_a:
            lines += [f"- 시간 예산으로 건너뛴 조합: {', '.join(skipped_a)}", ""]
        lines += ["## (A) 시험 기간 MAE (CO₂ ppm / 온도 ℃) — 보고용, 선택에는 쓰지 않음", "",
                  tab.to_markdown(index=False) if len(tab) else "(없음)", "",
                  "- 과제 3 본 모델(GRU 120분, 에폭 상한 40) 60분 CO₂: 실습실 31.8 / 사무실 40.1 / 회의실 52.3 "
                  "(현재값 유지 40.0 / 46.4 / 56.6) — reports/03_forecast.md", ""]
    lines += schedule_link()
    if loso is not None:
        t, ref, _, skipped_b = loso
        lines += [f"## (B) 한 공간 빼기 — {args.loso_arch.upper()} {args.loso_lookback}분", "",
                  "- 빼는 공간 데이터는 정규화 통계·학습·검증·α 어디에도 쓰지 않음. 그 공간 통계는 나머지 공간 평균, one-hot 은 0.",
                  "- '포함 학습' = 같은 설정으로 모든 공간을 넣어 학습한 모델의 같은 공간 시험 오차.", ""]
        if len(t):
            r60 = t[t["지평(분)"] == 60]
            lines += ["### 60분 요약", "", r60.to_markdown(index=False), "",
                      "### 전체 지평", "", t.to_markdown(index=False), ""]
        if skipped_b:
            lines += [f"- 시간 예산으로 건너뛴 공간: {', '.join(skipped_b)}", ""]
    lines += ["## 해석 (직접 작성)",
              "- 최적 조합과 근거(검증 점수, 학습 시간·파라미터 대비):",
              "- 과거 길이를 늘리거나 줄였을 때 지평별로 무엇이 달라졌나:",
              "- 구조별 차이(GRU·LSTM·TCN)와 Jetson 에서의 비용:",
              "- 일정 입력이 회의실 예측에 준 효과와 한계(예약표 부재):",
              "- 처음 보는 공간에서 현재값 유지보다 나은가, 포함 학습과 차이는 얼마인가:", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["arch", "loso"], help="(A) 구조×길이 또는 (B) 한 공간 빼기만")
    ap.add_argument("--full-grid", action="store_true", help="(A) 3종 × 1·2·4시간 9조합 전부")
    ap.add_argument("--epochs", type=int, default=20, help="(A) 에폭 상한")
    ap.add_argument("--patience", type=int, default=5, help="(A) 조기 종료")
    ap.add_argument("--loso-epochs", type=int, default=12, help="(B) 에폭 상한")
    ap.add_argument("--loso-patience", type=int, default=4, help="(B) 조기 종료")
    ap.add_argument("--loso-arch", choices=list(fc.ARCHS), default="gru")
    ap.add_argument("--loso-lookback", type=int, default=120, help="(B) 과거 길이(분)")
    ap.add_argument("--budget-min", type=float, default=110,
                    help="이 시간(분)을 넘기면 남은 학습을 건너뛰고 보고서를 쓴다")
    ap.add_argument("--no-schedule", action="store_true", help="일정 입력 없이 (기본은 schedule.yaml 사용)")
    args = ap.parse_args()

    from tsinfer.data.preprocess import clean, load_raw, to_wide

    t_start = time.time()
    deadline = t_start + args.budget_min * 60
    cfg = load_yaml("default.yaml")
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    schedule = None if args.no_schedule else load_yaml("schedule.yaml")
    print(f"데이터 {wide.index.min()} ~ {wide.index.max()}, 일정 입력 {'없음' if schedule is None else '있음'}, "
          f"예산 {args.budget_min:.0f}분")

    arch_res = stage2 = loso = None
    skipped_a: list[str] = []
    if args.only in (None, "arch"):
        arch_res, stage2, skipped_a = run_arch(cfg, wide, schedule, args.full_grid, args.epochs, args.patience,
                                               deadline)
    if args.only in (None, "loso"):
        loso = run_loso(cfg, wide, schedule, args.loso_arch, args.loso_lookback, args.loso_epochs,
                        args.loso_patience, deadline)
    text = write_report(cfg, wide, schedule, args, arch_res, stage2, skipped_a, loso, time.time() - t_start)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n저장: {REPORT} (전체 {(time.time() - t_start) / 60:.1f}분)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
