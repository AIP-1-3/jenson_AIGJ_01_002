"""과제 4: 이상 탐지 — 규칙 / 오토인코더 / 결합을 고장 5종으로 채점.

    python -m tsinfer.models.anomaly       # → models/anomaly_ae.pt, reports/04_anomaly.md, reports/figures/04_threshold.png

정답이 있는 시험을 만들려고 시험 기간 데이터에 고장을 일부러 넣는다 (공간 단위 5분 시계열).
  spike  : 한 칸 튐 (CO₂ +400ppm / 온도 +5℃ / 습도 +20%)
  stuck  : 3시간 같은 값 (센서 멈춤)
  drift  : 6시간에 걸쳐 서서히 어긋남 (CO₂ +250ppm, 온도 +3℃)
  noise  : 2시간 잡음 증가 (CO₂ σ60ppm, 온도 σ1℃)
  dropout: 1시간 값 없음 (수집 끊김)

채점 단위는 '경보 사건' (연속으로 켜진 칸을 하나로 묶음).
  탐지: 고장 구간 시작 ~ 끝+30분 안에 경보가 하나라도 켜짐 / 지연: 첫 경보 - 고장 시작
  헛경보: 어떤 고장 구간과도 겹치지 않는 경보 사건 수 ÷ 시험 일수 (전체 공간 합)
시험 기간 원본에 진짜 이상이 섞여 있으면 헛경보로 잡힌다 → 해석에서 사례로 확인.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.service.rules import SPIKE, STUCK_MIN

METRICS = ("co2", "temperature", "humidity")
VALID = {"co2": (350, 5000), "temperature": (-10, 50), "humidity": (0, 100)}
KINDS = ("spike", "stuck", "drift", "noise", "dropout")
DURATION_MIN = {"spike": 5, "stuck": 180, "drift": 360, "noise": 120, "dropout": 60}
GRACE = pd.Timedelta(minutes=30)
MODEL_PATH = ROOT / "models" / "anomaly_ae.pt"


# ---------------------------------------------------------------- 고장 주입

@dataclass
class Fault:
    kind: str
    space: str
    metric: str
    start: pd.Timestamp
    end: pd.Timestamp


def inject(wide: pd.DataFrame, spaces: list[str], per_kind: int = 15, seed: int = 0,
           step_min: int = 5) -> tuple[pd.DataFrame, list[Fault]]:
    """겹치지 않게 고장을 넣은 복사본과 정답 목록. 고장 사이는 최소 6시간 띄운다."""
    rng = np.random.default_rng(seed)
    w = wide.copy()
    faults: list[Fault] = []
    busy: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {sp: [] for sp in spaces}
    step = pd.Timedelta(minutes=step_min)
    sep = pd.Timedelta(hours=6)
    for kind in KINDS:
        n_steps = max(DURATION_MIN[kind] // step_min, 1)
        placed, tries = 0, 0
        while placed < per_kind and tries < per_kind * 200:
            tries += 1
            sp = spaces[rng.integers(len(spaces))]
            metric = "co2" if kind in ("spike", "drift", "noise") and rng.random() < 0.7 else \
                METRICS[rng.integers(len(METRICS))]
            if kind == "dropout":
                metric = "*"
            i0 = int(rng.integers(12, len(w) - n_steps - 12))
            t0, t1 = w.index[i0], w.index[i0] + n_steps * step
            if any(t0 < e + sep and t1 > s - sep for s, e in busy[sp]):
                continue
            cols = [(sp, m) for m in METRICS] if metric == "*" else [(sp, metric)]
            seg = w.loc[t0: t1 - step, cols]
            if seg.isna().mean().mean() > 0.1 or (metric != "*" and seg.std().iloc[0] == 0):
                continue                          # 원래 비어 있거나 원래 고정인 구간은 피한다
            idx = w.index[(w.index >= t0) & (w.index < t1)]
            if kind == "spike":
                w.loc[idx, (sp, metric)] += {"co2": 400.0, "temperature": 5.0, "humidity": 20.0}[metric]
            elif kind == "stuck":
                w.loc[idx, (sp, metric)] = w.loc[t0, (sp, metric)]
            elif kind == "drift":
                amp = {"co2": 250.0, "temperature": 3.0, "humidity": 12.0}[metric]
                w.loc[idx, (sp, metric)] += np.linspace(0, amp, len(idx))
            elif kind == "noise":
                sd = {"co2": 60.0, "temperature": 1.0, "humidity": 5.0}[metric]
                w.loc[idx, (sp, metric)] += rng.normal(0, sd, len(idx))
            elif kind == "dropout":
                w.loc[idx, cols] = np.nan
            busy[sp].append((t0, t1))
            faults.append(Fault(kind, sp, metric, t0, t1))
            placed += 1
    return w, faults


# ---------------------------------------------------------------- 규칙 (서비스 rules.py 와 같은 기준값)

NOISE_WIN = 12            # 잡음 규칙: 최근 1시간(12칸) 5분 변화량의 표준편차
NOISE_Q = 0.999           # 임계값 = 학습+검증(정상) 기간의 이 분위수 × NOISE_MARGIN, 공간·항목별
NOISE_MARGIN = 1.2


def diff_std(s: pd.Series, win: int = NOISE_WIN) -> pd.Series:
    return s.diff().rolling(win, min_periods=win // 2).std()


def fit_noise_thresholds(wide_normal: pd.DataFrame, spaces: list[str]) -> dict[tuple[str, str], float]:
    """시험 기간을 보지 않고 정상 기간에서만 잡음 규칙 임계값을 정한다."""
    out = {}
    for sp in spaces:
        for m in METRICS:
            if (sp, m) in wide_normal:
                q = diff_std(wide_normal[(sp, m)]).quantile(NOISE_Q)
                if pd.notna(q) and q > 0:
                    out[(sp, m)] = float(q) * NOISE_MARGIN
    return out


def rule_flags(wide: pd.DataFrame, spaces: list[str], step_min: int = 5,
               noise_thr: dict[tuple[str, str], float] | None = None) -> dict[str, pd.DataFrame]:
    """공간별 (시점 × 원인 항목) 불리언. 원인 이름: '<규칙>:<항목>'.
    noise_thr 가 있으면 '1시간 변화량 표준편차가 정상 기간 최고 수준을 넘는다' 잡음 규칙을 더한다."""
    out = {}
    for sp in spaces:
        f = pd.DataFrame(index=wide.index)
        allna = pd.Series(True, index=wide.index)
        for m in METRICS:
            if (sp, m) not in wide:
                continue
            s = wide[(sp, m)]
            allna &= s.isna()
            f[f"spike:{m}"] = (s.diff().abs() >= SPIKE[m]) | (s.diff(-1).abs() >= SPIKE[m])
            n = STUCK_MIN[m] // step_min
            r = s.rolling(n, min_periods=n)
            f[f"stuck:{m}"] = (r.max() - r.min()) == 0    # rolling std 는 긴 시계열에서 0 이 안 나올 수 있다
            lo, hi = VALID[m]
            f[f"range:{m}"] = s.notna() & ~s.between(lo, hi)
            if noise_thr and (sp, m) in noise_thr:
                f[f"noise:{m}"] = diff_std(s) > noise_thr[(sp, m)]
        f["gap:*"] = allna & allna.rolling(2, min_periods=2).sum().eq(2)   # 10분 이상 연속 결측
        out[sp] = f.fillna(False).astype(bool)
    return out


# ---------------------------------------------------------------- 오토인코더 (정상 구간만으로 학습)

CH = 2 * len(METRICS)   # 항목별 (값, 5분 변화량) — 변화량 채널이 잡음·급변을 크게 드러낸다


def ae_windows(wide: pd.DataFrame, sp: str, stats: dict[str, tuple[float, float]], L: int):
    """(N, L*6) 정규화 창과 창 끝 시점. 결측이 있는 창은 뺀다 (결측은 규칙이 맡는다)."""
    chans = []
    for m in METRICS:
        v = wide[(sp, m)].to_numpy()
        chans.append((v - stats[m][0]) / stats[m][1])
        chans.append(np.r_[np.nan, np.diff(v)] / stats[f"d_{m}"][1])
    arr = np.stack(chans, axis=1)
    if len(arr) < L:
        return np.zeros((0, L * CH), "float32"), wide.index[:0]
    win = np.lib.stride_tricks.sliding_window_view(arr, (L, CH))[:, 0]
    ok = ~np.isnan(win).any(axis=(1, 2))
    return win[ok].reshape(ok.sum(), -1).astype("float32"), wide.index[L - 1:][ok]


def build_ae(n_in: int):
    from torch import nn

    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(),
                         nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, n_in))


def train_ae(x: np.ndarray, epochs: int = 30, seed: int = 0, threads: int = 3):
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    model = build_ae(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x)
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 256):
            b = xt[perm[i:i + 256]]
            opt.zero_grad()
            loss = ((model(b) - b) ** 2).mean()
            loss.backward()
            opt.step()
    model.eval()
    return model


def ae_scores(model, x: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """창 점수(전체 MSE)와 항목별 MSE (N, 3) — 어느 센서가 원인인지 표시용.
    최근 쪽 칸에 가중치를 둬서 고장이 시작되자마자 점수가 오르게 한다."""
    import torch

    with torch.no_grad():
        rec = torch.cat([model(torch.from_numpy(x[i:i + 8192])) for i in range(0, len(x), 8192)]).numpy() \
            if len(x) else np.zeros_like(x)
    err = ((rec - x) ** 2).reshape(len(x), L, len(METRICS), 2)   # (값, 변화량) 채널을 항목별로 묶는다
    wts = np.linspace(0.5, 1.5, L)[None, :, None, None]
    per_metric = (err * wts).mean(axis=(1, 3))
    return per_metric.mean(axis=1), per_metric


@dataclass
class AE:
    model: Any
    stats: dict[str, tuple[float, float]]
    L: int
    threshold: float = np.inf

    def flags(self, wide: pd.DataFrame, sp: str, threshold: float | None = None) -> pd.DataFrame:
        thr = self.threshold if threshold is None else threshold
        x, t = ae_windows(wide, sp, self.stats, self.L)
        f = pd.DataFrame(False, index=wide.index, columns=[f"ae:{m}" for m in METRICS])
        if len(x):
            score, per = ae_scores(self.model, x, self.L)
            hit = score > thr
            cause = per.argmax(axis=1)
            for j, m in enumerate(METRICS):
                f.loc[t[hit & (cause == j)], f"ae:{m}"] = True
        return f

    def score_series(self, wide: pd.DataFrame, sp: str) -> np.ndarray:
        x, _ = ae_windows(wide, sp, self.stats, self.L)
        return ae_scores(self.model, x, self.L)[0] if len(x) else np.zeros(0)


def fit_ae(wide_train: pd.DataFrame, spaces: list[str], L: int = 12, epochs: int = 30, threads: int = 3) -> AE:
    stats = {}
    for m in METRICS:
        v = pd.concat([wide_train[(sp, m)] for sp in spaces]).dropna()
        stats[m] = (float(v.mean()), float(v.std()) or 1.0)
        d = pd.concat([wide_train[(sp, m)].diff() for sp in spaces]).dropna()
        stats[f"d_{m}"] = (0.0, float(d.std()) or 1.0)
    x = np.concatenate([ae_windows(wide_train, sp, stats, L)[0] for sp in spaces])
    return AE(train_ae(x, epochs=epochs, threads=threads), stats, L)


def pick_threshold(ae: AE, wide_valid: pd.DataFrame, spaces: list[str], max_alarms_per_day: float = 0.5) -> float:
    """정상(검증) 기간에서 경보 사건이 하루 max_alarms_per_day 이하가 되는 가장 낮은 임계값.
    목표(≤1회/일)의 절반으로 잡아 규칙 헛경보가 더해질 여유를 남긴다."""
    days = max((wide_valid.index[-1] - wide_valid.index[0]) / pd.Timedelta("1D"), 1e-9)
    scores = np.concatenate([ae.score_series(wide_valid, sp) for sp in spaces])
    for q in (0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999, 1.0):
        thr = float(np.quantile(scores, q)) if len(scores) else np.inf
        n = sum(len(events(ae.flags(wide_valid, sp, thr).any(axis=1))) for sp in spaces)
        if n / days <= max_alarms_per_day:
            return thr
    return float(scores.max()) if len(scores) else np.inf


# ---------------------------------------------------------------- 채점

def events(flag: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    flag = flag.fillna(False).astype(bool)
    if not flag.any():
        return []
    grp = (flag != flag.shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in flag[flag].groupby(grp[flag])]


def evaluate(flags: dict[str, pd.DataFrame], faults: list[Fault], days: float) -> pd.DataFrame:
    """고장 종류별 탐지율·지연, 그리고 헛경보/일. 원인 항목이 맞았는지도 센다."""
    rows = []
    for f in faults:
        fl = flags[f.space]
        win = fl[(fl.index >= f.start) & (fl.index <= f.end + GRACE)]
        hit = win.any(axis=1)
        first = hit.idxmax() if hit.any() else None
        cause_cols = list(win.columns) if f.metric == "*" else [c for c in win.columns if c.endswith(f":{f.metric}")]
        right_cause = bool(win[cause_cols].any().any()) if hit.any() else False
        rows.append({"kind": f.kind, "detected": hit.any(),
                     "delay_min": (first - f.start).total_seconds() / 60 if first is not None else np.nan,
                     "cause_ok": right_cause})
    fa = 0
    for sp, fl in flags.items():
        mine = [f for f in faults if f.space == sp]
        for s, e in events(fl.any(axis=1)):
            if not any(s <= f.end + GRACE and e >= f.start for f in mine):
                fa += 1
    det = pd.DataFrame(rows)
    if det.empty:
        return pd.DataFrame({"false_alarms_per_day": [round(fa / days, 2)]}, index=["전체"])
    out = det.groupby("kind").agg(n=("detected", "size"), detection_rate=("detected", "mean"),
                                  delay_min_median=("delay_min", "median"), cause_ok_rate=("cause_ok", "mean"))
    out.loc["전체"] = [len(det), det["detected"].mean(), det["delay_min"].median(), det["cause_ok"].mean()]
    out["false_alarms_per_day"] = round(fa / days, 2)
    out["detection_rate"] = (out["detection_rate"] * 100).round(1)
    out["cause_ok_rate"] = (out["cause_ok_rate"] * 100).round(1)
    return out.reindex(list(KINDS) + ["전체"])


def method_flags(method: str, wide: pd.DataFrame, spaces: list[str], ae: AE | None,
                 threshold: float | None = None,
                 noise_thr: dict[tuple[str, str], float] | None = None) -> dict[str, pd.DataFrame]:
    rules = rule_flags(wide, spaces, noise_thr=noise_thr)
    if method == "rules":
        return rules
    aef = {sp: ae.flags(wide, sp, threshold) for sp in spaces}
    if method == "ae":
        return aef
    return {sp: pd.concat([rules[sp], aef[sp]], axis=1) for sp in spaces}


def threshold_curve(ae: AE, wide_inj: pd.DataFrame, faults: list[Fault], spaces: list[str], days: float) -> pd.DataFrame:
    scores = np.concatenate([ae.score_series(wide_inj, sp) for sp in spaces])
    rows = []
    for q in (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995):
        thr = float(np.quantile(scores, q))
        r = evaluate(method_flags("ae", wide_inj, spaces, ae, thr), faults, days).loc["전체"]
        rows.append({"quantile": q, "threshold": round(thr, 3), "detection_rate": r["detection_rate"],
                     "false_alarms_per_day": r["false_alarms_per_day"]})
    return pd.DataFrame(rows)


def alarm_causes(flags: dict[str, pd.DataFrame], days: float, top: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """원본(고장 없음) 경보 사건을 원인 규칙별로 세고, 사건 목록 앞쪽을 돌려준다.
    원본 경보가 진짜 이상(수집 끊김 등)인지 헛경보인지 사람이 확인할 수 있게 하려는 것."""
    rows = []
    for sp, fl in flags.items():
        for s_, e in events(fl.any(axis=1)):
            seg = fl.loc[s_:e]
            causes = sorted(c for c in seg.columns if seg[c].any())
            rows.append({"space": sp, "start": s_, "end": e,
                         "minutes": int((e - s_).total_seconds() // 60) + 5, "causes": ", ".join(causes)})
    ev = pd.DataFrame(rows, columns=["space", "start", "end", "minutes", "causes"])
    if ev.empty:
        return pd.DataFrame(columns=["events", "per_day"]), ev
    kinds = ev["causes"].str.split(", ").explode()
    by = kinds.value_counts().rename("events").to_frame()
    by["per_day"] = (by["events"] / days).round(2)
    return by, ev.sort_values("start").head(top)


def run(cfg: dict[str, Any], wide: pd.DataFrame, per_kind: int = 15, epochs: int = 30, seed: int = 0):
    from tsinfer.data.split import split_by_time

    spaces = [sp for sp in cfg["spaces"] if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in METRICS)]
    train, valid, test = split_by_time(wide, cfg)
    ae = fit_ae(train, spaces, epochs=epochs, threads=cfg.get("model", {}).get("threads", 3))
    ae.threshold = pick_threshold(ae, valid, spaces)
    noise_thr = fit_noise_thresholds(pd.concat([train, valid]), spaces)
    test_inj, faults = inject(test, spaces, per_kind=per_kind, seed=seed)
    days = (test.index[-1] - test.index[0]) / pd.Timedelta("1D")
    results = {m: evaluate(method_flags(m, test_inj, spaces, ae, noise_thr=noise_thr), faults, days)
               for m in ("rules", "ae", "combined")}
    clean_flags = {m: method_flags(m, test, spaces, ae, noise_thr=noise_thr) for m in ("rules", "ae", "combined")}
    clean_fa = {m: evaluate(f, [], days)["false_alarms_per_day"].iloc[0] for m, f in clean_flags.items()}
    curve = threshold_curve(ae, test_inj, faults, spaces, days)
    diag = {"noise_thr": noise_thr, "clean_causes": alarm_causes(clean_flags["combined"], days)}
    return ae, spaces, results, clean_fa, curve, faults, (test.index[0], test.index[-1]), diag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-kind", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    import torch
    from tsinfer.data.preprocess import clean, load_raw, to_wide

    cfg = load_yaml("default.yaml")
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    ae, spaces, results, clean_fa, curve, faults, (t0, t1), diag = run(cfg, wide, args.per_kind, args.epochs)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": ae.model.state_dict(), "stats": ae.stats, "L": ae.L, "threshold": ae.threshold}, MODEL_PATH)

    figs = ROOT / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(curve["false_alarms_per_day"], curve["detection_rate"], marker="o")
        ax.axvline(1, color="gray", ls="--", lw=0.8)
        ax.axhline(80, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("false alarms / day")
        ax.set_ylabel("detection rate (%)")
        ax.set_title("Autoencoder threshold trade-off")
        fig.tight_layout()
        fig.savefig(figs / "04_threshold.png", dpi=120)
        plt.close(fig)
    except ImportError:
        pass

    lines = [
        "# 과제 4. 이상 탐지",
        "",
        "## 측정 조건",
        f"- 데이터: 실데이터, 학습(정상) ~{cfg['split']['train_end']}, 임계값 결정 = 검증 기간, 시험 {t0:%Y-%m-%d} ~ {t1:%Y-%m-%d} (UTC)",
        f"- 대상 공간: {', '.join(spaces)} / 항목: CO₂·온도·습도 / 5분 간격",
        f"- 고장 주입: 종류별 {args.per_kind}건 (seed 0), 지속 {DURATION_MIN} 분, 고장 사이 최소 6시간",
        f"- 오토인코더: 1시간(12칸)×3항목×(값, 5분 변화량) 창, 72→32→8→32→72, {args.epochs} 에폭, 임계값 {ae.threshold:.3f} (검증 기간 경보 ≤0.5회/일)",
        f"- 규칙: 급변 {SPIKE}, 값 고정 {STUCK_MIN}분, 범위 밖 {VALID}, 10분 이상 결측",
        f"- 잡음 규칙: 최근 1시간 5분 변화량 표준편차 > 정상(학습+검증) 기간 {NOISE_Q:.1%} 분위수 × {NOISE_MARGIN} "
        f"(공간·항목별, 시험 기간은 보지 않음): "
        + ", ".join(f"{sp}/{m} {v:.2f}" for (sp, m), v in diag["noise_thr"].items()),
        f"- 탐지 = 고장 시작~끝+{int(GRACE.total_seconds() // 60)}분 안의 경보, 헛경보 = 고장과 겹치지 않는 경보 사건 (전체 공간 합 / 일)",
        "",
        "## 결과 (탐지율 %, 지연 분, 원인 항목 일치 %, 헛경보 회/일)",
        "",
    ]
    for m, title in (("rules", "규칙"), ("ae", "오토인코더"), ("combined", "결합 (규칙 OR 오토인코더)")):
        lines += [f"### {title}", "", results[m].round(1).to_markdown(), ""]
    lines += [
        "## 고장을 넣지 않은 원본 시험 기간의 경보 (회/일)",
        "원본에도 진짜 이상이 있을 수 있어, 이 값이 0 이 아니면 경보 시각을 찾아 실제 사례로 확인한다.",
        "",
        pd.Series(clean_fa, name="false_alarms_per_day").to_frame().to_markdown(),
        "",
        "### 원본 경보의 원인 규칙 (결합 기준, 한 사건에 원인이 여럿이면 각각 셈)",
        "",
        diag["clean_causes"][0].to_markdown() if len(diag["clean_causes"][0]) else "(경보 없음)",
        "",
        "### 원본 경보 사건 (앞쪽 10건, UTC) — 실제 수집 끊김·센서 이상인지 확인",
        "",
        diag["clean_causes"][1].to_markdown(index=False) if len(diag["clean_causes"][1]) else "(경보 없음)",
        "",
        "## 임계값 - 헛경보 곡선 (오토인코더)",
        "",
        "![임계값 곡선](figures/04_threshold.png)",
        "",
        curve.to_markdown(index=False),
        "",
        "## 해석 (직접 작성)",
        "- 규칙이 잘 잡는 고장 / 모델이 잘 잡는 고장과 그 이유:",
        "- 결합했을 때 헛경보가 늘어난 만큼 탐지율이 올랐는가:",
        "- 운영 제안 (어떤 임계값, 어떤 경보를 누구에게):",
        "",
    ]
    out = ROOT / "reports" / "04_anomaly.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}, 모델: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
