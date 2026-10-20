"""과제 6: 사전학습 시계열 모델(Chronos-Bolt) zero-shot 비교.

    .venv/bin/pip install chronos-forecasting "numpy<2"   # 처음 한 번 (NumPy 1.26 유지)
    ~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.pretrained
    # 옵션: --model amazon/chronos-bolt-small(기본)|...-tiny|...-base  --device cpu|cuda  --every 60 --max-points 300

같은 시험 시점에서 현재값 유지 / GRU(과제 3, 있으면) / Chronos 를 비교한다.
  1) 정확도: 5·15·30·60분 MAE (Chronos 는 중앙값 예측)
  2) 구간 적중률: 실제값이 Chronos 10~90% 구간 안에 들어온 비율 (잘 맞춘 구간이면 약 80%)
  3) 추론 시간: 배치 1·전체 배치 지연 (워밍업 후 반복 평균, p95)
  4) 메모리: 프로세스 최대 RSS(+ GPU 면 최대 할당량)
학습은 하지 않는다 (zero-shot). 입력은 각 시점까지의 과거 CO₂·온도 한 줄씩(단변량)이다.
"""
from __future__ import annotations

import argparse
import resource
import time
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml

QUANTILES = [0.1, 0.5, 0.9]
TARGETS = ("co2", "temperature")


def eval_points(test_index: pd.DatetimeIndex, every_min: int, max_points: int, horizon_min: int) -> pd.DatetimeIndex:
    """시험 기간에서 every_min 간격으로 고른 기준 시점 (60분 뒤 정답이 있도록 끝은 뺀다)."""
    pts = test_index[test_index <= test_index[-1] - pd.Timedelta(minutes=horizon_min)]
    pts = pts[(pts - pts[0]) % pd.Timedelta(minutes=every_min) == pd.Timedelta(0)]
    if len(pts) > max_points:
        pts = pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]
    return pts


def contexts(s: pd.Series, points: pd.DatetimeIndex, length: int) -> np.ndarray:
    """(N, length) 과거 창. 시작 전·결측은 NaN (Chronos 가 결측으로 처리)."""
    pos = s.index.get_indexer(points)
    arr = s.to_numpy(dtype="float32")
    out = np.full((len(points), length), np.nan, "float32")
    for i, p in enumerate(pos):
        lo = max(0, p + 1 - length)
        out[i, length - (p + 1 - lo):] = arr[lo:p + 1]
    return out


def truth(s: pd.Series, points: pd.DatetimeIndex, horizons: list[int], step_min: int) -> np.ndarray:
    pos = s.index.get_indexer(points)
    arr = s.to_numpy()
    return np.stack([arr[np.clip(pos + h, 0, len(arr) - 1)] for h in horizons], axis=1)


def peak_mem_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # 리눅스: KB


def load_chronos(name: str, device: str):
    import torch
    from chronos import BaseChronosPipeline

    return BaseChronosPipeline.from_pretrained(name, device_map=device, torch_dtype=torch.float32)


def chronos_predict(pipe, ctx: np.ndarray, H: int, batch: int = 64) -> np.ndarray:
    """(N, H, 3) 분위수 예측."""
    import torch

    outs = []
    for i in range(0, len(ctx), batch):
        q, _ = pipe.predict_quantiles(torch.from_numpy(ctx[i:i + batch]), prediction_length=H,
                                      quantile_levels=QUANTILES)
        outs.append(q.float().cpu().numpy())
    return np.concatenate(outs)


def latency(pipe, ctx: np.ndarray, H: int, batch_sizes=(1, 4), repeat: int = 30, warmup: int = 5,
            device: str = "cpu") -> pd.DataFrame:
    import torch

    rows = []
    for b in batch_sizes:
        x = torch.from_numpy(ctx[:b])
        ts = []
        for i in range(warmup + repeat):
            t0 = time.perf_counter()
            pipe.predict_quantiles(x, prediction_length=H, quantile_levels=QUANTILES)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                ts.append((time.perf_counter() - t0) * 1000)
        rows.append({"batch": b, "mean_ms": round(float(np.mean(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
                     "repeat": repeat, "warmup": warmup})
    return pd.DataFrame(rows)


def gru_at(wide: pd.DataFrame, points: pd.DatetimeIndex) -> tuple[dict[tuple[str, str], np.ndarray], str] | None:
    """과제 3 모델이 있으면 같은 시점 예측. 없거나 창이 비면 None/NaN."""
    from tsinfer.models import forecast as fc

    if not fc.MODEL_PATH.exists():
        return None
    model, spec = fc.load()
    x, _, info = fc.make_windows(wide, spec, at=points, with_target=False)
    if not len(x):
        return None
    pred = fc.decode(fc.predict_array(model, x), info, spec)
    out = {}
    for sp in spec.spaces:
        for m in TARGETS:
            arr = np.full((len(points), len(spec.horizons)), np.nan)
            sel = (info["space"] == sp).to_numpy()
            pos = points.get_indexer(pd.DatetimeIndex(info.loc[sel, "time"]))
            arr[pos] = pred[m][sel]
            out[(sp, m)] = arr
    return out, "gru"


def compare(wide: pd.DataFrame, points: pd.DatetimeIndex, spaces: list[str], predict_fn, ctx_len: int,
            H: int, step_min: int, report_minutes: list[int], gru=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """같은 시점 MAE 표와 구간 적중률 표."""
    horizons = list(range(1, H + 1))
    rows, cov_rows = [], []
    for sp in spaces:
        for m in TARGETS:
            s = wide[(sp, m)]
            y = truth(s, points, horizons, step_min)
            cur = s.reindex(points).to_numpy()
            q = predict_fn(contexts(s, points, ctx_len), H)
            g = gru[(sp, m)] if gru and (sp, m) in gru else None
            for hm in report_minutes:
                h = hm // step_min - 1
                ok = ~np.isnan(y[:, h]) & ~np.isnan(cur)
                if g is not None:
                    ok &= ~np.isnan(g[:, h])
                if not ok.any():
                    continue
                r = {"space": sp, "metric": m, "horizon_min": hm, "n": int(ok.sum()),
                     "persistence": float(np.abs(cur[ok] - y[ok, h]).mean()),
                     "chronos": float(np.abs(q[ok, h, 1] - y[ok, h]).mean())}
                if g is not None:
                    r["gru"] = float(np.abs(g[ok, h] - y[ok, h]).mean())
                rows.append(r)
                inside = (y[ok, h] >= q[ok, h, 0]) & (y[ok, h] <= q[ok, h, 2])
                cov_rows.append({"space": sp, "metric": m, "horizon_min": hm,
                                 "coverage_10_90_%": round(100 * float(inside.mean()), 1),
                                 "mean_width": round(float((q[ok, h, 2] - q[ok, h, 0]).mean()), 2)})
    return pd.DataFrame(rows), pd.DataFrame(cov_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--context-hours", type=int, default=42, help="과거 입력 길이 (기본 42시간 = 504칸)")
    ap.add_argument("--every", type=int, default=60, help="시험 기간에서 몇 분마다 평가할지")
    ap.add_argument("--max-points", type=int, default=300)
    args = ap.parse_args()

    import torch

    from tsinfer.data.preprocess import clean, load_raw, to_wide
    from tsinfer.data.split import split_by_time

    torch.set_num_threads(load_yaml("default.yaml").get("model", {}).get("threads", 3))
    cfg = load_yaml("default.yaml")
    step = cfg["forecast"]["step_minutes"]
    H = cfg["forecast"]["horizon_minutes"] // step
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    _, _, test = split_by_time(wide, cfg)
    spaces = [sp for sp in cfg["spaces"] if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in TARGETS)]
    points = eval_points(test.index, args.every, args.max_points, cfg["forecast"]["horizon_minutes"])
    ctx_len = args.context_hours * 60 // step

    mem0 = peak_mem_mb()
    t0 = time.perf_counter()
    pipe = load_chronos(args.model, args.device)
    load_s = time.perf_counter() - t0
    res, cov = compare(wide, points, spaces, lambda c, h: chronos_predict(pipe, c, h), ctx_len, H, step,
                       cfg["forecast"]["report_horizons"], gru=(g := gru_at(wide, points)) and g[0])
    lat = latency(pipe, contexts(wide[(spaces[0], "co2")], points, ctx_len), H, device=args.device)
    mem = {"process_peak_rss_mb": round(peak_mem_mb(), 0), "rss_before_load_mb": round(mem0, 0)}
    if args.device == "cuda":
        mem["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 0)
    n_params = sum(p.numel() for p in pipe.inner_model.parameters()) if hasattr(pipe, "inner_model") else None

    methods = [c for c in ("persistence", "gru", "chronos") if c in res]
    table = res.pivot_table(index=["space", "metric"], columns="horizon_min", values=methods).round(2)
    lines = [
        "# 과제 6. 사전학습 모델 비교 (zero-shot)",
        "",
        "## 측정 조건",
        f"- 모델: {args.model} (파라미터 {n_params:,}개)" if n_params else f"- 모델: {args.model}",
        f"- 장치: Jetson Orin Nano Super, {args.device}, 전력 모드 MAXN_SUPER, torch {torch.__version__}, "
        f"CPU 스레드 {torch.get_num_threads()}, gpu-run.sh 락 안에서 실행",
        f"- 입력: 기준 시점까지 과거 {args.context_hours}시간({ctx_len}칸) 단변량, 출력 5~60분 12칸 분위수 {QUANTILES}",
        f"- 평가 시점: 시험 기간 {test.index.min():%Y-%m-%d}~{test.index.max():%Y-%m-%d} 중 {args.every}분 간격 "
        f"{len(points)}개 (같은 시점에서 세 방법 비교)",
        f"- 모델 불러오기 {load_s:.1f}s",
        "",
        "## 1. 정확도 MAE (CO₂ ppm / 온도 ℃)",
        "",
        table.to_markdown(),
        "",
        "## 2. 10~90% 구간 적중률 (기대값 80%)",
        "",
        cov.pivot_table(index=["space", "metric"], columns="horizon_min", values="coverage_10_90_%").to_markdown(),
        "",
        "## 3. 추론 시간 (12칸 예측 한 번)",
        "",
        lat.to_markdown(index=False),
        "",
        "## 4. 메모리",
        "",
        pd.Series(mem).to_frame("MB").to_markdown(),
        "",
        "## 채택 의견 (직접 작성)",
        "- 정확도: GRU·현재값 유지 대비 어디서 낫고 어디서 못한가:",
        "- 구간 예측이 서비스에 주는 가치 (경보 신뢰도 표시 등):",
        "- 다음 주 로컬 LLM 과 같은 8GB 보드에 함께 올릴 수 있는가 (메모리·지연):",
        "- 결론: 채택 / 보조로만 / 미채택 과 이유:",
        "",
    ]
    out = ROOT / "reports" / "06_pretrained.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
