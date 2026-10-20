"""과제 7: 엣지 최적화 — PyTorch CPU/GPU · ONNX Runtime · TensorRT FP16 지연과 변환 오차.

    .venv/bin/pip install onnx onnxruntime "numpy<2"        # 처음 한 번
    ~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.edge
    # → models/*.onnx, models/*.engine, reports/07_edge.md

대상 모델
- 예측 GRU(과제 3): 변환 전후 CO₂ 예측 차이 (목표 ≤1ppm)
- 이상 탐지 오토인코더(과제 4): 임계값 판정이 바뀐 비율 (목표 ≤0.1%) — '분류 변화'로 본다

측정
- 배치 1·4, 워밍업 후 반복 평균·p95 지연 (TensorRT 는 trtexec 의 Latency = 입출력 복사 포함)
- 측정하는 동안 tegrastats 로 RAM·전력(VDD_IN)·GPU 사용률을 같이 기록
- 측정 조건: 전력 모드는 바꾸지 않고(MAXN_SUPER), 클럭 고정(jetson_clocks) 안 함 — 공용 장비 규칙
PyTorch GPU 는 Jetson 용 torch(CUDA) 가 있을 때만 잰다. PyPI CPU 빌드면 '미측정'으로 남긴다.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml

MODELS = ROOT / "models"
TRTEXEC_CANDIDATES = ("trtexec", "/usr/src/tensorrt/bin/trtexec")


# ---------------------------------------------------------------- 내보내기

def export_onnx(model, sample: np.ndarray, path: Path, input_name: str = "x") -> Path:
    import torch

    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.from_numpy(sample[:1]), str(path), input_names=[input_name], output_names=["y"],
                      dynamic_axes={input_name: {0: "batch"}, "y": {0: "batch"}}, opset_version=17, dynamo=False)
    return path


def torch_run(model, x: np.ndarray, device: str = "cpu") -> np.ndarray:
    import torch

    m = model.to(device)
    with torch.no_grad():
        out = np.concatenate([m(torch.from_numpy(x[i:i + 1024]).to(device)).cpu().numpy()
                              for i in range(0, len(x), 1024)])
    model.to("cpu")
    return out


def ort_session(path: Path, threads: int = 3):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def ort_run(sess, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: x[i:i + 1024]})[0] for i in range(0, len(x), 1024)])


# ---------------------------------------------------------------- 지연

def time_fn(fn, repeat: int, warmup: int) -> dict[str, float]:
    ts = []
    for i in range(warmup + repeat):
        t0 = time.perf_counter()
        fn()
        if i >= warmup:
            ts.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": round(float(np.mean(ts)), 3), "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def find_trtexec() -> str | None:
    for c in TRTEXEC_CANDIDATES:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


_LAT = re.compile(r"Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms, "
                  r"percentile\(95%\) = ([\d.]+) ms")


def parse_trtexec_latency(text: str) -> dict[str, float] | None:
    """trtexec 성능 요약의 'Latency:' 줄 (입출력 복사 포함)."""
    m = None
    for m in _LAT.finditer(text):
        pass
    if m is None:
        return None
    return {"mean_ms": float(m.group(3)), "p95_ms": float(m.group(5))}


def parse_trtexec_output(path: Path, name: str = "y") -> np.ndarray:
    """--exportOutput JSON → (batch, …) 배열."""
    data = json.loads(path.read_text())
    item = next(d for d in data if d["name"] == name)
    dims = [int(v) for v in str(item["dimensions"]).split("x")]
    return np.asarray(item["values"], dtype="float32").reshape(dims)


def trt_build_and_measure(trtexec: str, onnx: Path, sample: np.ndarray, batches: list[int], check_batch: int,
                          workdir: Path, input_name: str = "x") -> tuple[dict[int, dict[str, float]], np.ndarray, str]:
    """FP16 엔진을 만들고 배치별 지연을 잰 뒤, check_batch 개 입력의 출력을 돌려준다."""
    shape = "x".join(str(d) for d in sample.shape[1:])
    engine = onnx.with_suffix(".engine")
    lo, hi = min(batches + [check_batch]), max(batches + [check_batch])
    build = [trtexec, f"--onnx={onnx}", "--fp16", f"--saveEngine={engine}",
             f"--minShapes={input_name}:{lo}x{shape}", f"--optShapes={input_name}:{max(batches)}x{shape}",
             f"--maxShapes={input_name}:{hi}x{shape}", "--skipInference"]
    subprocess.run(build, check=True, capture_output=True, text=True)
    lat = {}
    log = ""
    for b in batches:
        r = subprocess.run([trtexec, f"--loadEngine={engine}", f"--shapes={input_name}:{b}x{shape}",
                            "--warmUp=500", "--iterations=200", "--duration=0", "--percentile=95"],
                           check=True, capture_output=True, text=True)
        log += r.stdout[-1500:]
        lat[b] = parse_trtexec_latency(r.stdout) or {"mean_ms": np.nan, "p95_ms": np.nan}
    inp = workdir / f"{onnx.stem}_in.bin"
    out = workdir / f"{onnx.stem}_out.json"
    np.ascontiguousarray(sample[:check_batch], dtype="float32").tofile(inp)
    subprocess.run([trtexec, f"--loadEngine={engine}", f"--shapes={input_name}:{check_batch}x{shape}",
                    f"--loadInputs={input_name}:{inp}", f"--exportOutput={out}", "--iterations=1", "--warmUp=0",
                    "--duration=0"], check=True, capture_output=True, text=True)
    return lat, parse_trtexec_output(out), log


# ---------------------------------------------------------------- 장비 기록

class Tegrastats:
    """측정하는 동안 tegrastats 출력을 모아 RAM·전력·GPU 사용률 요약. 없으면 조용히 빈 값."""

    RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    GPU = re.compile(r"GR3D_FREQ (\d+)%")
    PWR = re.compile(r"VDD_IN (\d+)mW")

    def __init__(self, interval_ms: int = 200):
        self.interval = interval_ms
        self.lines: list[str] = []
        self.proc = None

    def __enter__(self):
        if shutil.which("tegrastats"):
            self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval)],
                                         stdout=subprocess.PIPE, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)

    def __exit__(self, *_):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)

    def summary(self) -> dict[str, float]:
        def vals(rx):
            return [float(m.group(1)) for line in self.lines if (m := rx.search(line))]
        ram, gpu, pwr = vals(self.RAM), vals(self.GPU), vals(self.PWR)
        return {"samples": len(self.lines),
                "ram_used_max_mb": max(ram) if ram else np.nan,
                "gpu_util_mean_%": round(float(np.mean(gpu)), 1) if gpu else np.nan,
                "power_vdd_in_mean_mw": round(float(np.mean(pwr)), 0) if pwr else np.nan,
                "power_vdd_in_max_mw": max(pwr) if pwr else np.nan}


# ---------------------------------------------------------------- 대상 모델과 입력

def gru_target(wide: pd.DataFrame, test_index: pd.DatetimeIndex):
    from tsinfer.models import forecast as fc

    model, spec = fc.load()
    x, _, info = fc.make_windows(wide, spec, at=test_index, with_target=False)

    def to_ppm(out: np.ndarray) -> np.ndarray:
        return fc.decode(out, info.iloc[:len(out)].reset_index(drop=True), spec)["co2"]
    return model, x, to_ppm


def ae_target(wide: pd.DataFrame, test_index: pd.DatetimeIndex):
    import torch

    from tsinfer.models import anomaly as an

    ckpt = torch.load(an.MODEL_PATH, map_location="cpu", weights_only=False)
    L = ckpt["L"]
    model = an.build_ae(L * an.CH)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    test = wide.loc[wide.index.isin(test_index)]
    spaces = [sp for sp in wide.columns.get_level_values(0).unique()
              if all((sp, m) in wide and wide[(sp, m)].notna().any() for m in an.METRICS)]
    x = np.concatenate([an.ae_windows(test, sp, ckpt["stats"], L)[0] for sp in spaces])

    def flags(out: np.ndarray, inp: np.ndarray) -> np.ndarray:
        err = ((out - inp[:len(out)]) ** 2).reshape(len(out), L, len(an.METRICS), 2)
        score = (err * np.linspace(0.5, 1.5, L)[None, :, None, None]).mean(axis=(1, 3)).mean(axis=1)
        return score > ckpt["threshold"]
    return model, x, flags


# ---------------------------------------------------------------- 실행

def measure(name: str, model, x: np.ndarray, batches: list[int], repeat: int, warmup: int, threads: int,
            trtexec: str | None, check_n: int, workdir: Path):
    """방법별 지연 표와 방법별 출력(check_n 개) 묶음."""
    import torch

    torch.set_num_threads(threads)
    onnx = export_onnx(model, x, MODELS / f"{name}.onnx")
    sess = ort_session(onnx, threads)
    xs = x[:check_n]
    outputs = {"pytorch_cpu": torch_run(model, xs), "onnxruntime_cpu": ort_run(sess, xs)}
    rows = []
    for b in batches:
        xb = torch.from_numpy(x[:b])
        with torch.no_grad():
            rows.append({"method": "pytorch_cpu", "batch": b, **time_fn(lambda: model(xb), repeat, warmup)})
        rows.append({"method": "onnxruntime_cpu", "batch": b,
                     **time_fn(lambda: sess.run(None, {"x": x[:b]}), repeat, warmup)})
    if torch.cuda.is_available():
        gm = model.to("cuda")
        outputs["pytorch_gpu"] = torch_run(model, xs, "cuda")
        for b in batches:
            xb = torch.from_numpy(x[:b]).cuda()

            def step():
                with torch.no_grad():
                    gm(xb)
                torch.cuda.synchronize()
            rows.append({"method": "pytorch_gpu", "batch": b, **time_fn(step, repeat, warmup)})
        model.to("cpu")
    else:
        rows += [{"method": "pytorch_gpu", "batch": b, "mean_ms": np.nan, "p95_ms": np.nan, "note": "CUDA torch 없음"}
                 for b in batches]
    if trtexec:
        lat, out, _ = trt_build_and_measure(trtexec, onnx, x, batches, len(xs), workdir)
        outputs["tensorrt_fp16"] = out.reshape(len(xs), -1)
        rows += [{"method": "tensorrt_fp16", "batch": b, **lat[b]} for b in batches]
    else:
        rows += [{"method": "tensorrt_fp16", "batch": b, "mean_ms": np.nan, "p95_ms": np.nan, "note": "trtexec 없음"}
                 for b in batches]
    return pd.DataFrame(rows), outputs, onnx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--check", type=int, default=2048, help="변환 오차를 비교할 시험 창 수")
    args = ap.parse_args()

    import torch

    from tsinfer.data.preprocess import clean, load_raw, to_wide
    from tsinfer.data.split import split_by_time

    cfg = load_yaml("default.yaml")
    threads = cfg.get("model", {}).get("threads", 3)
    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    _, _, test = split_by_time(wide, cfg)
    trtexec = find_trtexec()
    workdir = MODELS / "edge_tmp"
    workdir.mkdir(parents=True, exist_ok=True)
    batches = [1, 4]
    sections, lat_tables, hw = [], [], {}

    with Tegrastats() as tg:
        model, x, to_ppm = gru_target(wide, test.index)
        lat, outs, _ = measure("forecast_gru", model, x, batches, args.repeat, args.warmup, threads, trtexec,
                               min(args.check, len(x)), workdir)
        ref = to_ppm(outs["pytorch_cpu"])
        diff = {k: float(np.abs(to_ppm(v) - ref).max()) for k, v in outs.items() if k != "pytorch_cpu"}
        lat_tables.append(lat.assign(model="forecast_gru"))
        sections += ["### 예측 GRU: PyTorch CPU 대비 CO₂ 예측 최대 차이 (ppm, 목표 ≤1)", "",
                     pd.Series(diff, name="max_abs_diff_ppm").round(4).to_frame().assign(
                         **{"pass(≤1ppm)": lambda d: d["max_abs_diff_ppm"] <= 1}).to_markdown(), ""]
        try:
            model, x, flags = ae_target(wide, test.index)
            lat, outs, _ = measure("anomaly_ae", model, x, batches, args.repeat, args.warmup, threads, trtexec,
                                   min(args.check * 4, len(x)), workdir)
            n = len(outs["pytorch_cpu"])
            ref = flags(outs["pytorch_cpu"], x)
            change = {k: round(100 * float((flags(v, x) != ref).mean()), 4) for k, v in outs.items() if k != "pytorch_cpu"}
            lat_tables.append(lat.assign(model="anomaly_ae"))
            sections += [f"### 오토인코더: 이상 판정이 바뀐 비율 (%, {n}개 창, 목표 ≤0.1%)", "",
                         pd.Series(change, name="flag_change_%").to_frame().assign(
                             **{"pass(≤0.1%)": lambda d: d["flag_change_%"] <= 0.1}).to_markdown(), ""]
        except FileNotFoundError:
            sections += ["### 오토인코더", "", "models/anomaly_ae.pt 가 없어 건너뜀 (과제 4 먼저 실행)", ""]
    hw = tg.summary()
    shutil.rmtree(workdir, ignore_errors=True)

    lat_all = pd.concat(lat_tables, ignore_index=True)
    table = lat_all.pivot_table(index=["model", "method"], columns="batch", values=["mean_ms", "p95_ms"], dropna=False)
    lines = [
        "# 과제 7. 엣지 최적화",
        "",
        "## 측정 조건",
        f"- 장비: Jetson Orin Nano Super 8GB, JetPack 7.2.1, 전력 모드 MAXN_SUPER(변경 안 함), 클럭 고정 안 함, "
        f"gpu-run.sh 락 안에서 단독 측정",
        f"- torch {torch.__version__} (CUDA {'있음' if torch.cuda.is_available() else '없음'}), CPU 스레드 {threads}, "
        f"onnxruntime CPU, TensorRT FP16 ({'trtexec' if trtexec else '없음'})",
        f"- 반복 {args.repeat}회, 워밍업 {args.warmup}회 (TensorRT 는 워밍업 500ms + 200회), 배치 {batches}",
        "- 입력: 실데이터 시험 기간 창. TensorRT 지연은 입출력 복사 포함(trtexec Latency)",
        "",
        "## 지연 (ms)",
        "",
        table.round(3).to_markdown(),
        "",
        "## 변환 전후 차이",
        "",
        *sections,
        "## 측정 중 장비 기록 (tegrastats)",
        "",
        pd.Series(hw).to_frame("value").to_markdown(),
        "",
        "## 해석 (직접 작성)",
        "- 배치 1 에서 GPU 가 CPU 보다 느리거나 비슷하다면 그 이유 (작은 모델, 복사 비용):",
        "- 서비스(매분 1회, 공간 3개)에 실제로 필요한 속도와 비교한 결론:",
        "- FP16 오차가 경보 판단에 주는 영향:",
        "",
    ]
    out = ROOT / "reports" / "07_edge.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
