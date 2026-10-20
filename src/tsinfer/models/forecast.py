"""과제 3: GRU 로 5~60분 CO₂·온도 예측.

    python -m tsinfer.models.forecast                 # 학습 → models/forecast_gru.pt, reports/03_forecast.md
    python -m tsinfer.models.forecast --compare-time  # 시각·요일 입력을 뺀 모델도 학습해 비교

설계
- 입력: 최근 lookback(기본 2시간) 동안 그 공간의 CO₂·온도·습도, 외기(사무실 밖) 온도·습도,
  한국 시간 시각(sin/cos)·평일 여부, 공간 표시(one-hot). 정규화 통계는 학습 기간으로만 계산.
- 출력: 5~60분 12칸을 한 번에(직접 다중 출력). 반복 예측은 오차가 쌓인다.
- 목표값은 '지금 값 대비 변화량'. 현재값 유지가 가장 강한 기준선이라, 모델은 그 위에 얹을 변화만 배운다.
- 공간마다 따로가 아니라 CO₂ 가 있는 공간 전체로 하나를 학습한다 (데이터가 3배).
- 일정표가 없어 '일정 포함/제외' 대신 '시각·요일 포함/제외'를 비교한다 (일정의 대리 변수).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml

TARGETS = ("co2", "temperature")
OWN = ("co2", "temperature", "humidity")
OUTDOOR = ("temperature", "humidity")
MODEL_PATH = ROOT / "models" / "forecast_gru.pt"
SCHEDULE_OFFSETS = (0, 15, 30, 60)   # 지금, 15·30·60분 뒤 일정상 사용 중인지 (일정은 미리 알 수 있는 정보)


@dataclass
class FeatureSpec:
    """학습·서비스가 같은 입력을 만들도록 모델 파일에 같이 저장한다."""

    spaces: list[str]
    outdoor_space: str | None
    lookback: int                       # 칸 수
    horizons: list[int]                 # 칸 수 (1..12)
    step_minutes: int
    timezone: str
    time_features: bool = True
    holidays: list[str] = field(default_factory=list)      # "YYYY-MM-DD" (한국 시간) — 평일 여부에서 뺀다
    schedule: dict[str, Any] | None = None                  # configs/schedule.yaml — 있으면 일정 입력 사용
    alpha: dict[str, list[float]] = field(default_factory=dict)  # 항목 → 지평별 변화량 축소 계수 (검증으로 결정)
    mean: dict[str, float] = field(default_factory=dict)   # "공간|항목" → 평균
    std: dict[str, float] = field(default_factory=dict)
    target_std: dict[str, float] = field(default_factory=dict)   # "공간|항목" → 변화량 표준편차

    @property
    def n_features(self) -> int:
        return (len(OWN) + len(OUTDOOR) + (3 if self.time_features else 0)
                + (len(SCHEDULE_OFFSETS) if self.schedule else 0) + len(self.spaces))

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def model_spaces(wide: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """CO₂·온도가 모두 있는 공간만 예측 대상 (사무실 밖은 CO₂ 없음 → 외기 입력)."""
    return [sp for sp in cfg["spaces"]
            if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in TARGETS)]


def fit_spec(wide: pd.DataFrame, train_index: pd.DatetimeIndex, cfg: dict[str, Any],
             time_features: bool = True, schedule: dict[str, Any] | None = None) -> FeatureSpec:
    mc, fc = cfg["model"], cfg["forecast"]
    step = fc["step_minutes"]
    spaces = model_spaces(wide, cfg)
    outdoor = mc.get("outdoor_space")
    if outdoor not in cfg["spaces"]:
        outdoor = None
    spec = FeatureSpec(
        spaces=spaces, outdoor_space=outdoor,
        lookback=mc["lookback_minutes"] // step,
        horizons=list(range(1, fc["horizon_minutes"] // step + 1)),
        step_minutes=step, timezone=cfg.get("timezone", "Asia/Seoul"), time_features=time_features,
        holidays=[str(d) for d in cfg.get("holidays", [])], schedule=schedule,
    )
    tr = wide.loc[wide.index.isin(train_index)]
    cols = [(sp, m) for sp in spaces for m in OWN] + ([(outdoor, m) for m in OUTDOOR] if outdoor else [])
    for sp, m in cols:
        if (sp, m) not in tr:
            continue
        s = tr[(sp, m)]
        spec.mean[f"{sp}|{m}"] = float(s.mean()) if s.notna().any() else 0.0
        spec.std[f"{sp}|{m}"] = float(s.std()) if s.notna().sum() > 1 and s.std() > 0 else 1.0
    for sp in spaces:
        for m in TARGETS:
            d = (tr[(sp, m)].shift(-spec.horizons[-1]) - tr[(sp, m)]).std()
            spec.target_std[f"{sp}|{m}"] = float(d) if np.isfinite(d) and d > 0 else 1.0
    return spec


def is_workday(index: pd.DatetimeIndex, spec: FeatureSpec) -> np.ndarray:
    """평일이고 공휴일이 아닌 시점 (한국 시간)."""
    local = index.tz_convert(spec.timezone)
    hol = np.isin(local.strftime("%Y-%m-%d"), spec.holidays) if spec.holidays else np.zeros(len(index), bool)
    return (local.dayofweek < 5) & ~hol


def space_features(wide: pd.DataFrame, space: str, spec: FeatureSpec) -> pd.DataFrame:
    """한 공간의 시점별 입력 특성 (정규화 끝난 값). 자기 공간 값의 NaN 은 그대로 둔다 (창에서 걸러냄)."""
    def z(sp: str, m: str) -> pd.Series:
        s = wide[(sp, m)] if (sp, m) in wide else pd.Series(np.nan, index=wide.index)
        return (s - spec.mean.get(f"{sp}|{m}", 0.0)) / spec.std.get(f"{sp}|{m}", 1.0)

    f = pd.DataFrame(index=wide.index)
    for m in OWN:
        f[m] = z(space, m)
    for m in OUTDOOR:   # 외기는 빠져도 예측을 막지 않게 0(학습 평균)으로 채운다
        f[f"out_{m}"] = z(spec.outdoor_space, m).fillna(0.0) if spec.outdoor_space else 0.0
    f["humidity"] = f["humidity"].fillna(0.0)
    if spec.time_features:
        local = wide.index.tz_convert(spec.timezone)
        hour = local.hour + local.minute / 60
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        f["weekday"] = is_workday(wide.index, spec).astype(float)
    if spec.schedule:
        from tsinfer.models.state import occupancy_labels

        for off in SCHEDULE_OFFSETS:
            idx = wide.index + pd.Timedelta(minutes=off)
            occ = occupancy_labels(idx, [space], spec.schedule, tz=spec.timezone)[space].to_numpy()
            occ = np.where(np.isnan(occ), 0.0, occ) * is_workday(idx, spec)
            f[f"sched_{off}"] = occ
    for sp in spec.spaces:
        f[f"is_{sp}"] = float(sp == space)
    return f.astype("float32")


def make_windows(wide: pd.DataFrame, spec: FeatureSpec, at: pd.DatetimeIndex | None = None,
                 with_target: bool = True) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """(N, lookback, F) 입력, (N, 2*H) 변화량 목표(정규화), 행별 정보(time, space, 현재값).

    at: 예측 기준 시점(창의 마지막 칸) 후보. None 이면 전체.
    입력 창 안에 자기 공간 CO₂·온도 결측이 있거나, 목표에 결측이 있으면 버린다.
    """
    L, H = spec.lookback, len(spec.horizons)
    xs, ys, infos = [], [], []
    for sp in spec.spaces:
        f = space_features(wide, sp, spec)
        arr = f.to_numpy()
        if len(arr) < L:
            continue
        win = np.lib.stride_tricks.sliding_window_view(arr, (L, arr.shape[1]))[:, 0]   # (T-L+1, L, F)
        end_idx = np.arange(L - 1, len(arr))
        own_ok = ~np.isnan(win[:, :, :2]).any(axis=(1, 2))
        mask = own_ok
        if at is not None:
            mask &= wide.index[end_idx].isin(at)
        cur = {m: wide[(sp, m)].to_numpy() for m in TARGETS}
        if with_target:
            fut = np.stack([
                np.stack([np.r_[cur[m][h:], np.full(h, np.nan)] for h in spec.horizons], axis=1)
                for m in TARGETS
            ], axis=1)                                                       # (T, 2, H)
            delta = fut - np.stack([cur[m] for m in TARGETS], axis=1)[:, :, None]
            delta = delta / np.array([spec.target_std[f"{sp}|{m}"] for m in TARGETS])[None, :, None]
            delta = delta[end_idx].reshape(len(end_idx), -1)
            mask &= ~np.isnan(delta).any(axis=1)
            ys.append(delta[mask].astype("float32"))
        xs.append(win[mask].astype("float32"))
        infos.append(pd.DataFrame({"time": wide.index[end_idx][mask], "space": sp,
                                   **{f"cur_{m}": cur[m][end_idx][mask] for m in TARGETS}}))
    if not xs:
        return np.zeros((0, L, spec.n_features), "float32"), None, pd.DataFrame()
    x = np.concatenate(xs)
    y = np.concatenate(ys) if with_target else None
    return x, y, pd.concat(infos, ignore_index=True)


def decode(pred: np.ndarray, info: pd.DataFrame, spec: FeatureSpec) -> dict[str, np.ndarray]:
    """정규화 변화량 → 실제 값. 반환: 항목별 (N, H)."""
    H = len(spec.horizons)
    out = {}
    for i, m in enumerate(TARGETS):
        scale = info["space"].map(lambda sp: spec.target_std[f"{sp}|{m}"]).to_numpy()[:, None]
        a = np.asarray(spec.alpha.get(m, [1.0] * H))[None, :]
        out[m] = info[f"cur_{m}"].to_numpy()[:, None] + pred[:, i * H:(i + 1) * H] * scale * a
    return out


# ---------------------------------------------------------------- torch 부분 (학습·서비스에서만 import)

def build_model(spec: FeatureSpec, mc: dict[str, Any]):
    import torch
    from torch import nn

    class GRUForecast(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(spec.n_features, mc["hidden"], num_layers=mc["layers"], batch_first=True,
                              dropout=mc["dropout"] if mc["layers"] > 1 else 0.0)
            self.head = nn.Sequential(nn.Dropout(mc["dropout"]), nn.Linear(mc["hidden"], len(TARGETS) * len(spec.horizons)))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, h = self.gru(x)
            return self.head(h[-1])

    return GRUForecast()


def train(x_tr, y_tr, x_va, y_va, spec: FeatureSpec, mc: dict[str, Any], log=print):
    import torch

    torch.manual_seed(mc["seed"])
    np.random.seed(mc["seed"])
    torch.set_num_threads(mc.get("threads", 3))
    model = build_model(spec, mc)
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"])
    loss_fn = torch.nn.L1Loss()   # 평가 지표(MAE)와 같은 손실
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv, yv = torch.from_numpy(x_va), torch.from_numpy(y_va)
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, mc["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), mc["batch_size"]):
            idx = perm[i:i + mc["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = sum(loss_fn(model(xv[i:i + 4096]), yv[i:i + 4096]).item() * len(xv[i:i + 4096])
                      for i in range(0, len(xv), 4096)) / max(len(xv), 1)
        history.append({"epoch": epoch, "train": total / len(xt), "valid": val, "sec": time.perf_counter() - t0})
        log(f"epoch {epoch:2d} train {total / len(xt):.4f} valid {val:.4f} ({history[-1]['sec']:.1f}s)")
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= mc["patience"]:
                log(f"조기 종료: {mc['patience']}번 연속 개선 없음")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_array(model, x: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i + 4096])).numpy() for i in range(0, len(x), 4096)]) \
            if len(x) else np.zeros((0, 0), "float32")


def save(model, spec: FeatureSpec, mc: dict[str, Any], path: Path = MODEL_PATH, extra: dict | None = None) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "spec": spec.to_dict(), "model_cfg": mc, **(extra or {})}, path)


def load(path: Path = MODEL_PATH):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    spec = FeatureSpec(**ckpt["spec"])
    model = build_model(spec, ckpt["model_cfg"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, spec


# ---------------------------------------------------------------- 평가·보고

def score(pred: dict[str, np.ndarray], info: pd.DataFrame, wide: pd.DataFrame, spec: FeatureSpec,
          report_minutes: list[int], method: str) -> pd.DataFrame:
    """공간·항목·지평별 MAE. 같은 시점의 현재값 유지 MAE 도 함께 낸다."""
    rows = []
    for m in TARGETS:
        for hm in report_minutes:
            hi = hm // spec.step_minutes - 1
            truth = np.array([wide[(sp, m)].get(t + pd.Timedelta(minutes=hm), np.nan)
                              for t, sp in zip(info["time"], info["space"])])
            ok = ~np.isnan(truth)
            for sp in spec.spaces:
                sel = ok & (info["space"].to_numpy() == sp)
                if not sel.any():
                    continue
                rows.append({"space": sp, "metric": m, "horizon_min": hm, "n": int(sel.sum()),
                             method: float(np.abs(pred[m][sel, hi] - truth[sel]).mean()),
                             "persistence": float(np.abs(info[f"cur_{m}"].to_numpy()[sel] - truth[sel]).mean())})
    return pd.DataFrame(rows)


def fit_alpha(pred: np.ndarray, info: pd.DataFrame, wide: pd.DataFrame, spec: FeatureSpec) -> dict[str, list[float]]:
    """지평별로 '현재값 + α·예측 변화량' 의 MAE 가 가장 작은 α (0 = 현재값 유지, 1 = 모델 그대로).
    검증 기간에서 정하므로, 그 기간에서는 현재값 유지보다 나빠지지 않는다."""
    grid = np.round(np.arange(0, 1.21, 0.1), 2)
    spec_raw = FeatureSpec(**{**spec.to_dict(), "alpha": {}})
    raw = decode(pred, info, spec_raw)
    out = {}
    for m in TARGETS:
        cur = info[f"cur_{m}"].to_numpy()[:, None]
        delta = raw[m] - cur
        alphas = []
        for k, h in enumerate(spec.horizons):
            truth = np.array([wide[(sp, m)].get(t + pd.Timedelta(minutes=h * spec.step_minutes), np.nan)
                              for t, sp in zip(info["time"], info["space"])])
            ok = ~np.isnan(truth)
            if not ok.any():
                alphas.append(1.0)
                continue
            maes = [np.abs(cur[ok, 0] + a * delta[ok, k] - truth[ok]).mean() for a in grid]
            alphas.append(float(grid[int(np.argmin(maes))]))
        out[m] = alphas
    return out


def run(cfg: dict[str, Any], wide: pd.DataFrame, time_features: bool = True, log=print,
        schedule: dict[str, Any] | None = None):
    from tsinfer.data.split import split_by_time

    train_df, valid_df, test_df = split_by_time(wide, cfg)
    spec = fit_spec(wide, train_df.index, cfg, time_features=time_features, schedule=schedule)
    # 검증은 일하는 날만: 연휴(예: 추석)만 남은 검증 기간은 '빈방'만 보고 모델을 고르게 된다
    va_index = valid_df.index[is_workday(valid_df.index, spec)]
    if len(va_index) < len(valid_df) * 0.2:
        va_index = valid_df.index
    x_tr, y_tr, _ = make_windows(wide, spec, train_df.index)
    x_va, y_va, info_va = make_windows(wide, spec, va_index)
    x_te, _, info_te = make_windows(wide, spec, test_df.index, with_target=False)
    log(f"창 개수 train {len(x_tr):,} / valid(일하는 날) {len(x_va):,} / test {len(x_te):,}, "
        f"입력 {spec.lookback}칸×{spec.n_features}")
    model, history = train(x_tr, y_tr, x_va, y_va, spec, cfg["model"], log=log)
    spec.alpha = fit_alpha(predict_array(model, x_va), info_va, wide, spec)
    log(f"축소 계수 α: {spec.alpha}")
    t0 = time.perf_counter()
    pred = decode(predict_array(model, x_te), info_te, spec)
    infer_ms = (time.perf_counter() - t0) / max(len(x_te), 1) * 1000
    res = score(pred, info_te, wide, spec, cfg["forecast"]["report_horizons"], "gru")
    return model, spec, res, history, infer_ms


def report_table(res: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    t = res.pivot_table(index=["space", "metric"], columns="horizon_min", values=methods + ["persistence"])
    return t.round(2)


def target_check(res: pd.DataFrame, method: str = "gru") -> pd.DataFrame:
    r = res[(res["metric"] == "co2") & (res["horizon_min"] == 60)].copy()
    r["improve_%"] = ((1 - r[method] / r["persistence"]) * 100).round(1)
    r["pass(≥30%)"] = r["improve_%"] >= 30
    t = res[(res["metric"] == "temperature")].groupby("space").apply(
        lambda g: bool((g[method] < g["persistence"]).all()), include_groups=False)
    r["temp_all_horizons_better"] = r["space"].map(t)
    return r[["space", "persistence", method, "improve_%", "pass(≥30%)", "temp_all_horizons_better"]].round(2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare-time", action="store_true", help="시각·요일 입력을 뺀 모델도 학습해 비교")
    ap.add_argument("--schedule", action="store_true",
                    help="configs/schedule.yaml 일정을 입력에 넣고, 일정을 뺀 모델과 비교 (과제 3 '일정 포함/제외')")
    args = ap.parse_args()

    from tsinfer.data.preprocess import clean, load_raw, to_wide

    cfg = load_yaml("default.yaml")
    raw = load_raw(metrics=cfg["metrics"])
    wide = to_wide(clean(raw, cfg)[0], cfg)
    sched = load_yaml("schedule.yaml") if args.schedule else None
    if sched and sched.get("source") == "placeholder":
        print("⚠ schedule.yaml 이 임시 가정(placeholder)입니다. 실제 시간표를 넣어야 일정 효과를 제대로 잴 수 있습니다.")
    model, spec, res, history, infer_ms = run(cfg, wide, schedule=sched)
    save(model, spec, cfg["model"], extra={"trained_at": pd.Timestamp.now(tz="UTC").isoformat()})
    methods = ["gru"]

    def add_variant(label: str, **kw):
        _, _, r, _, _ = run(cfg, wide, **kw)
        nonlocal res
        res = res.merge(r[["space", "metric", "horizon_min", "gru"]].rename(columns={"gru": label}),
                        on=["space", "metric", "horizon_min"], how="left")
        methods.append(label)

    if args.compare_time:
        add_variant("gru_no_time", time_features=False, schedule=sched)
    if sched:
        add_variant("gru_no_schedule", schedule=None)

    from tsinfer.data.split import split_by_time
    _, _, test_df = split_by_time(wide, cfg)
    mc = cfg["model"]
    lines = [
        "# 과제 3. 60분 예측 (GRU)",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터 InfluxDB iot-bucket + 수집기, 학습 ~{cfg['split']['train_end']}, 검증 ~{cfg['split']['valid_end']}, "
        f"시험 {test_df.index.min():%Y-%m-%d} ~ {test_df.index.max():%Y-%m-%d} (UTC)",
        f"- 대상 공간: {', '.join(spec.spaces)} (외기 입력: {spec.outdoor_space})",
        f"- 입력 {mc['lookback_minutes']}분({spec.lookback}칸) × {spec.n_features}특성, 출력 5~60분 12칸 한 번에, 목표 = 현재값 대비 변화량",
        f"- GRU hidden {mc['hidden']}, layers {mc['layers']}, dropout {mc['dropout']}, Adam lr {mc['lr']}, batch {mc['batch_size']}, "
        f"에폭 {len(history)}/{mc['epochs']}(조기 종료 patience {mc['patience']}, 검증은 일하는 날만), L1 손실, seed {mc['seed']}",
        f"- 공휴일 {len(spec.holidays)}일은 평일에서 제외, 일정 입력: {'있음' if spec.schedule else '없음'}",
        f"- 장비: Jetson Orin Nano Super, PyTorch CPU {mc.get('threads', 3)}스레드, 평균 에폭 "
        f"{np.mean([h['sec'] for h in history]):.1f}s, 시험 추론 {infer_ms:.3f}ms/창(배치 4096)",
        "",
        "## 결과 MAE (같은 시험 시점, CO₂ ppm / 온도 ℃)",
        "",
        report_table(res, methods).to_markdown(),
        "",
        "## 목표 확인: 60분 CO₂ 현재값 유지 대비 −30%, 온도도 기준선보다 우수",
        "",
        target_check(res).to_markdown(index=False),
        "",
    ]
    if args.compare_time:
        r60 = res[res["horizon_min"] == 60][["space", "metric", "gru", "gru_no_time"]].copy()
        r60["시각·요일 효과"] = (r60["gru_no_time"] - r60["gru"]).round(2)
        lines += ["## 시각·요일 입력 포함/제외 (60분)", "", r60.round(2).to_markdown(index=False),
                  "", "→ \"시각·요일을 넣으면 60분 CO₂ 오차가 ○ppm 줄어든다\" 로 해석에 옮긴다.", ""]
    if sched:
        r60 = res[res["horizon_min"] == 60][["space", "metric", "gru", "gru_no_schedule"]].copy()
        r60["일정 효과"] = (r60["gru_no_schedule"] - r60["gru"]).round(2)
        lines += [f"## 일정 입력 포함/제외 (60분, schedule source={sched.get('source')})", "",
                  r60.round(2).to_markdown(index=False), "",
                  "→ \"일정을 넣으면 60분 CO₂ 오차가 ○ppm 줄어든다\" 로 해석에 옮긴다.", ""]
    lines += [f"## 변화량 축소 계수 α (검증 기간 일하는 날에서 결정, 0=현재값 유지 1=모델 그대로)", "",
              pd.DataFrame(spec.alpha, index=[h * spec.step_minutes for h in spec.horizons]).T.to_markdown(), ""]
    lines += ["## 학습 곡선", "", pd.DataFrame(history).round(4).to_markdown(index=False), "",
              "## 해석 (직접 작성)", "- 공간별 오차 차이와 원인:", "- 오차가 큰 시간대(입실·퇴실 직후 등):",
              "- 입력 길이·구조를 바꿔 본 결과:", ""]
    out = ROOT / "reports" / "03_forecast.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    res.to_csv(ROOT / "reports" / "03_forecast_mae.csv", index=False)
    print("\n".join(lines[:40]))
    print(f"\n저장: {out}, 모델: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
