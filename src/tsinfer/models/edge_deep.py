"""과제 7 심화: INT8 변환 + 모델 4개 동시 적재 메모리와 로컬 LLM 동시 적재 조합.

    ~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.edge_deep
    # → reports/07b_edge_deep.md (엔진·임시 파일은 models/edge_deep_tmp 에 만들고 끝나면 지운다)
    # 옵션: --skip-int8  --skip-memory  --calib-n 512  --check 2048  --chronos-model amazon/chronos-bolt-small

(a) INT8: 시험 이전(학습+검증) 기간 창 일부를 보정 데이터로 써서 GRU·오토인코더 INT8 엔진을 만든다.
    1순위: TensorRT 파이썬 + IInt8EntropyCalibrator2 (보정 입력은 ctypes 로 libcudart 에 직접 복사 — CUDA torch 가 없어서)
    2순위: onnxruntime 정적 양자화로 Q/DQ ONNX 를 만든 뒤 trtexec --int8 (명시적 양자화)
    둘 다 실패하면 사유를 표에 남긴다. GRU 의 반복(RNN) 층은 INT8 이 안 돼 FP16 으로 남을 수 있다 → 층별 정밀도를 센다.
    비교: FP16 엔진 대비 지연, PyTorch CPU 대비 CO₂ 최대 차이(ppm)·판정(환기 경보 / 이상 판정) 변화율.
(b) 메모리: 한 프로세스에 torch → GRU → 오토인코더 → 상태 분류기 → Chronos-Bolt 를 차례로 올리며
    RSS 증가분과 보드 여유(MemAvailable = free -m 의 available)를 기록하고, 8GB 보드에서 로컬 LLM 과의 조합을 제안한다.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from tsinfer.config import ROOT, load_yaml
from tsinfer.models import edge
from tsinfer.models.pretrained import rss_mb

WORKDIR = edge.MODELS / "edge_deep_tmp"
# 가정: 로컬 LLM (llama.cpp 등 4bit 양자화) 가중치 + KV 캐시·런타임 대략값. 실측 아님.
LLM_ASSUMED_MB = {"LLM 1B 4bit (가정 ≈1.2GB)": 1200, "LLM 3B 4bit (가정 ≈2.5GB)": 2500}
HEADROOM_OK_MB = 1500      # 이만큼 남으면 '여유', 500 이상이면 '빠듯', 그 아래는 '부족' (스왑 들어가면 지연이 튄다)
HEADROOM_TIGHT_MB = 500


# ---------------------------------------------------------------- 메모리 측정 도구

def mem_available_mb(meminfo: str | None = None) -> float:
    """/proc/meminfo 의 MemAvailable (free -m 의 available 와 같은 값)."""
    try:
        text = meminfo if meminfo is not None else Path("/proc/meminfo").read_text()
    except OSError:
        return float("nan")
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def load_steps(steps: list[tuple[str, Callable[[], Any]]]) -> tuple[pd.DataFrame, list[Any]]:
    """단계마다 (불러오기 + 추론 한 번) 을 실행하고 RSS·MemAvailable 변화를 잰다. 불러온 객체는 끝까지 붙잡아 둔다."""
    keep, rows = [], []
    for name, fn in steps:
        gc.collect()
        r0, a0, t0 = rss_mb(), mem_available_mb(), time.perf_counter()
        try:
            obj = fn()
            keep.append(obj)
            note = "" if obj is not None else "건너뜀"
        except Exception as e:   # 파일 없음·패키지 없음 등은 표에 남기고 다음 단계
            obj, note = None, f"실패: {type(e).__name__}: {str(e)[:100]}"
        gc.collect()
        r1, a1 = rss_mb(), mem_available_mb()
        ok = obj is not None
        rows.append({"step": name, "rss_increase_mb": round(r1 - r0, 1) if ok else np.nan,
                     "rss_after_mb": round(r1, 0), "mem_available_after_mb": round(a1, 0),
                     "available_drop_mb": round(a0 - a1, 0) if ok else np.nan,
                     "sec": round(time.perf_counter() - t0, 1), "note": note})
    return pd.DataFrame(rows), keep


def suggest_combos(component_mb: dict[str, float], combos: dict[str, list[str]], available_mb: float,
                   llms: dict[str, float] = LLM_ASSUMED_MB) -> pd.DataFrame:
    """조합별 필요 메모리와 남는 여유. available_mb = 측정 시작 때 보드 여유 (이 프로세스 기본분은 이미 빠진 값)."""
    rows = []
    for cname, parts in combos.items():
        missing = [p for p in parts if not np.isfinite(component_mb.get(p, np.nan))]
        need = float(sum(component_mb.get(p, 0.0) for p in parts if p not in missing))
        for lname, lmb in {"LLM 없음": 0.0, **llms}.items():
            left = available_mb - need - lmb
            verdict = "여유" if left >= HEADROOM_OK_MB else "빠듯" if left >= HEADROOM_TIGHT_MB else "부족"
            rows.append({"combo": cname, "llm": lname, "models_mb": round(need, 0), "llm_mb": lmb,
                         "left_mb": round(left, 0), "verdict": verdict,
                         "note": ("측정 못 한 구성: " + ", ".join(missing)) if missing else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (b) 동시 적재

def _state_model_path(cfg: dict[str, Any]) -> tuple[Path | None, str]:
    """학습해 둔 상태 분류기. 없으면 state.run 으로 학습해 임시 파일에 저장 (데이터는 측정 전에 내린다)."""
    p = edge.MODELS / "state.joblib"
    if p.exists():
        return p, "models/state.joblib"
    import joblib

    from tsinfer.data.preprocess import clean, load_raw, to_wide
    from tsinfer.models import state as st

    wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
    sched = load_yaml("schedule.yaml")
    _, fitted, thresholds, _, feats, _, _ = st.run(cfg, wide, sched)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    tmp = WORKDIR / "state_tmp.joblib"
    joblib.dump({"models": fitted, "thresholds": thresholds, "features": feats}, tmp)
    del wide, fitted
    gc.collect()
    return tmp, "state.run 으로 새로 학습 (임시)"


def memory_steps(cfg: dict[str, Any], chronos_model: str, state_path: Path | None) -> list[tuple[str, Callable]]:
    threads = cfg.get("model", {}).get("threads", 3)

    def torch_import():
        import torch

        torch.set_num_threads(threads)
        return torch

    def gru_torch():
        import torch

        from tsinfer.models import forecast as fc

        model, spec = fc.load()
        with torch.no_grad():
            model(torch.zeros(1, spec.lookback, spec.n_features))
        return model

    def onnx_sess(name: str):
        def fn():
            p = edge.MODELS / f"{name}.onnx"
            if not p.exists():
                return None                           # edge.py 를 먼저 돌리면 생긴다
            sess = edge.ort_session(p, threads)
            shp = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]
            sess.run(None, {sess.get_inputs()[0].name: np.zeros(shp, "float32")})
            return sess
        return fn

    def ae_torch():
        import torch

        from tsinfer.models import anomaly as an

        ckpt = torch.load(an.MODEL_PATH, map_location="cpu", weights_only=False)
        model = an.build_ae(ckpt["L"] * an.CH)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            model(torch.zeros(1, ckpt["L"] * an.CH))
        return model

    def state_clf():
        if state_path is None:
            return None
        import joblib

        bundle = joblib.load(state_path)
        for key, m in bundle["models"].items():
            fx = bundle["features"]["occupied" if key.startswith("occupied") else "ventilation"]
            m.predict_proba(pd.DataFrame(np.zeros((1, len(fx))), columns=fx))
        return bundle

    def chronos():
        import torch

        from tsinfer.models import pretrained as pt

        pipe = pt.load_chronos(chronos_model, "cpu")
        pipe.predict_quantiles(torch.zeros(1, 504) + 500, prediction_length=12, quantile_levels=pt.QUANTILES)
        return pipe

    return [("torch import", torch_import), ("GRU (torch)", gru_torch), ("GRU (ONNX Runtime)", onnx_sess("forecast_gru")),
            ("오토인코더 (torch)", ae_torch), ("오토인코더 (ONNX Runtime)", onnx_sess("anomaly_ae")),
            ("상태 분류기 (sklearn)", state_clf), (f"Chronos-Bolt ({chronos_model.split('/')[-1]})", chronos)]


def default_combos(steps: pd.DataFrame) -> tuple[dict[str, float], dict[str, list[str]]]:
    comp = dict(zip(steps["step"], steps["rss_increase_mb"].astype(float)))
    chronos = next((s for s in comp if s.startswith("Chronos")), "Chronos-Bolt")
    combos = {
        "서비스 ONNX (GRU·오토인코더 ONNX + 상태 분류기, torch 없이)": ["GRU (ONNX Runtime)", "오토인코더 (ONNX Runtime)",
                                                                  "상태 분류기 (sklearn)"],
        "서비스 torch (torch + GRU + 오토인코더 + 상태 분류기)": ["torch import", "GRU (torch)", "오토인코더 (torch)",
                                                         "상태 분류기 (sklearn)"],
        "서비스 torch + Chronos-Bolt": ["torch import", "GRU (torch)", "오토인코더 (torch)", "상태 분류기 (sklearn)", chronos],
    }
    return comp, combos


# ---------------------------------------------------------------- (a) INT8

class CudaRT:
    """libcudart 를 ctypes 로 직접 부른다 (CUDA torch·pycuda 가 없는 Jetson 용). 없으면 OSError."""

    def __init__(self):
        cands = [ctypes.util.find_library("cudart"), "/usr/local/cuda/lib64/libcudart.so",
                 "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so"]
        for c in cands:
            if c:
                try:
                    self.lib = ctypes.CDLL(c)
                    break
                except OSError:
                    continue
        else:
            raise OSError("libcudart 를 찾지 못함")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]

    def malloc(self, nbytes: int) -> int:
        p = ctypes.c_void_p()
        if self.lib.cudaMalloc(ctypes.byref(p), nbytes) != 0:
            raise RuntimeError("cudaMalloc 실패")
        return int(p.value)

    def h2d(self, dev: int, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        if self.lib.cudaMemcpy(dev, arr.ctypes.data, arr.nbytes, 1) != 0:   # 1 = HostToDevice
            raise RuntimeError("cudaMemcpy 실패")

    def free(self, dev: int) -> None:
        self.lib.cudaFree(dev)


def build_int8_tensorrt(onnx: Path, calib: np.ndarray, batches: list[int], check_batch: int, engine: Path,
                        calib_batch: int = 32, input_name: str = "x") -> Path:
    """TensorRT 파이썬 + 엔트로피 보정으로 INT8(+FP16 대체) 엔진. 실패하면 예외 (사유가 메시지)."""
    import tensorrt as trt

    if not hasattr(trt, "IInt8EntropyCalibrator2"):
        raise RuntimeError(f"TensorRT {trt.__version__} 에 IInt8EntropyCalibrator2 없음 (암시적 양자화 제거됨)")
    cuda = CudaRT()
    data = np.ascontiguousarray(calib, dtype="float32")
    cache = engine.with_suffix(".calib")

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.dev = cuda.malloc(data[:calib_batch].nbytes)

        def get_batch_size(self):
            return calib_batch

        def get_batch(self, names):
            if self.i + calib_batch > len(data):
                return None
            cuda.h2d(self.dev, data[self.i:self.i + calib_batch])
            self.i += calib_batch
            return [self.dev]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, c):
            cache.write_bytes(bytes(c))

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError("ONNX 읽기 실패: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)        # INT8 이 안 되는 층(RNN 등)은 FP16 으로
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    shape = tuple(calib.shape[1:])
    lo, hi = min(batches + [check_batch]), max(batches + [check_batch])
    prof = builder.create_optimization_profile()
    prof.set_shape(input_name, (lo, *shape), (max(batches), *shape), (hi, *shape))
    config.add_optimization_profile(prof)
    cprof = builder.create_optimization_profile()
    cprof.set_shape(input_name, (calib_batch, *shape), (calib_batch, *shape), (calib_batch, *shape))
    config.set_calibration_profile(cprof)
    calibrator = Calib()
    config.int8_calibrator = calibrator
    try:
        ser = builder.build_serialized_network(network, config)
    finally:
        cuda.free(calibrator.dev)
    if ser is None:
        raise RuntimeError("build_serialized_network 가 None (INT8 미지원 층 또는 보정 실패 — TensorRT 로그 확인)")
    engine.write_bytes(bytes(ser))
    return engine


def quantize_qdq(onnx: Path, calib: np.ndarray, out: Path, input_name: str = "x", batch: int = 64) -> Path:
    """onnxruntime 정적 양자화 → Q/DQ ONNX (대칭 INT8, TensorRT 가 읽는 형식)."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{input_name: np.ascontiguousarray(calib[i:i + batch], dtype="float32")}
                            for i in range(0, len(calib), batch)])

        def get_next(self):
            return next(self.it, None)

    quantize_static(str(onnx), str(out), Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
    return out


def int8_layer_share(layer_info: Path | str | dict | list) -> tuple[int, int]:
    """trtexec --exportLayerInfo JSON 에서 (INT8 로 도는 층 수, 전체 층 수). 상세 정보가 없으면 (0, 층 수)."""
    data = layer_info
    if isinstance(layer_info, (str, Path)):
        data = json.loads(Path(layer_info).read_text())
    layers = data.get("Layers", []) if isinstance(data, dict) else data
    n8 = sum(1 for ly in layers if isinstance(ly, dict) and "int8" in json.dumps(ly).lower())
    return n8, len(layers)


def layer_info(trtexec: str, engine: Path, sample: np.ndarray, workdir: Path, input_name: str = "x") -> tuple[int, int]:
    shape = "x".join(str(d) for d in sample.shape[1:])
    out = workdir / f"{engine.stem}_layers.json"
    subprocess.run([trtexec, f"--loadEngine={engine}", f"--shapes={input_name}:1x{shape}", "--iterations=1",
                    "--warmUp=0", "--duration=0", "--profilingVerbosity=detailed", f"--exportLayerInfo={out}"],
                   check=True, capture_output=True, text=True)
    return int8_layer_share(out)


def _short(e: Exception) -> str:
    msg = getattr(e, "stderr", None) or str(e)
    msg = " ".join(str(msg).split())
    return f"{type(e).__name__}: {msg[-200:]}"


def int8_for_model(name: str, model, x: np.ndarray, calib: np.ndarray, batches: list[int], check_n: int,
                   trtexec: str | None, repeat: int, warmup: int, threads: int, workdir: Path):
    """한 모델의 INT8 시도. 반환: 지연 행 목록, 방법별 출력(check_n 개), 시도 기록 행 목록."""
    import torch

    torch.set_num_threads(threads)
    xs = x[:check_n]
    onnx = edge.export_onnx(model, x, workdir / f"{name}.onnx")
    outputs = {"pytorch_cpu": edge.torch_run(model, xs)}
    lat, tries = [], []
    sess32 = edge.ort_session(onnx, threads)      # CPU 쪽 비교 기준 (onnxruntime FP32)
    outputs["onnxruntime_cpu"] = edge.ort_run(sess32, xs)
    for b in batches:
        lat.append({"method": "onnxruntime_cpu", "batch": b,
                    **edge.time_fn(lambda: sess32.run(None, {"x": x[:b]}), repeat, warmup)})

    # 기준: FP16 엔진 (edge.py 와 같은 방법)
    if trtexec:
        try:
            eng = edge.trt_build(trtexec, onnx, x, batches, check_n, engine=workdir / f"{name}_fp16.engine",
                                 extra=("--profilingVerbosity=detailed",))
            l16, out, _ = edge.trt_measure(trtexec, eng, x, batches, check_n, workdir)
            outputs["tensorrt_fp16"] = out.reshape(len(xs), -1)
            lat += [{"method": "tensorrt_fp16", "batch": b, **l16[b]} for b in batches]
        except Exception as e:
            tries.append({"model": name, "path": "TensorRT FP16 (기준)", "result": "실패 — " + _short(e)})
    else:
        tries.append({"model": name, "path": "TensorRT FP16 (기준)", "result": "trtexec 없음 — 측정 안 함"})

    # 1순위: TensorRT 파이썬 + 엔트로피 보정
    eng8 = None
    try:
        eng8 = build_int8_tensorrt(onnx, calib, batches, check_n, workdir / f"{name}_int8.engine")
        tries.append({"model": name, "path": "TensorRT 파이썬 IInt8EntropyCalibrator2", "result": "엔진 생성 성공"})
        label = "tensorrt_int8_calib"
    except Exception as e:
        tries.append({"model": name, "path": "TensorRT 파이썬 IInt8EntropyCalibrator2", "result": "실패 — " + _short(e)})

    # 2순위: Q/DQ ONNX (onnxruntime 정적 양자화) → onnxruntime INT8 CPU, trtexec --int8
    qdq = None
    try:
        qdq = quantize_qdq(onnx, calib, workdir / f"{name}_qdq.onnx")
        sess = edge.ort_session(qdq, threads)
        outputs["onnxruntime_int8_qdq"] = edge.ort_run(sess, xs)
        for b in batches:
            lat.append({"method": "onnxruntime_int8_qdq", "batch": b,
                        **edge.time_fn(lambda: sess.run(None, {"x": x[:b]}), repeat, warmup)})
        tries.append({"model": name, "path": "onnxruntime 정적 양자화 (Q/DQ ONNX)", "result": "성공"})
    except Exception as e:
        tries.append({"model": name, "path": "onnxruntime 정적 양자화 (Q/DQ ONNX)", "result": "실패 — " + _short(e)})
    if eng8 is None and qdq is not None and trtexec:
        try:
            eng8 = edge.trt_build(trtexec, qdq, x, batches, check_n, precision=("--int8", "--fp16"),
                                  engine=workdir / f"{name}_int8.engine", extra=("--profilingVerbosity=detailed",))
            tries.append({"model": name, "path": "trtexec --int8 (Q/DQ ONNX)", "result": "엔진 생성 성공"})
            label = "tensorrt_int8_qdq"
        except Exception as e:
            tries.append({"model": name, "path": "trtexec --int8 (Q/DQ ONNX)", "result": "실패 — " + _short(e)})
    elif eng8 is None:
        tries.append({"model": name, "path": "trtexec --int8 (Q/DQ ONNX)",
                      "result": "시도 안 함 — " + ("trtexec 없음" if not trtexec else "Q/DQ ONNX 없음")})

    if eng8 is not None:
        if not trtexec:
            tries.append({"model": name, "path": "INT8 엔진 측정", "result": "trtexec 없음 — 지연·출력 측정 불가"})
        else:
            try:
                l8, out, _ = edge.trt_measure(trtexec, eng8, x, batches, check_n, workdir)
                outputs[label] = out.reshape(len(xs), -1)
                lat += [{"method": label, "batch": b, **l8[b]} for b in batches]
                n8, n = layer_info(trtexec, eng8, x, workdir)
                tries.append({"model": name, "path": "INT8 엔진 층 정밀도",
                              "result": f"INT8 로 도는 층 {n8}/{n} (나머지는 FP16/FP32 — RNN 등 INT8 미지원 층)"})
            except Exception as e:
                tries.append({"model": name, "path": "INT8 엔진 측정", "result": "실패 — " + _short(e)})
    return lat, outputs, tries


def accuracy_gru(outputs: dict[str, np.ndarray], to_ppm, threshold: float) -> pd.DataFrame:
    """PyTorch CPU 대비 CO₂ 최대·평균 차이(ppm)와 환기 판정(60분 안 예측 ≥ 임계값) 변화율."""
    ref = to_ppm(outputs["pytorch_cpu"])
    ref_flag = (ref >= threshold).any(axis=1)
    rows = {}
    for k, v in outputs.items():
        if k == "pytorch_cpu":
            continue
        p = to_ppm(v)
        rows[k] = {"max_abs_diff_ppm": round(float(np.abs(p - ref).max()), 4),
                   "mean_abs_diff_ppm": round(float(np.abs(p - ref).mean()), 4),
                   "vent_flag_change_%": round(100 * float(((p >= threshold).any(axis=1) != ref_flag).mean()), 4)}
    return pd.DataFrame(rows).T


def accuracy_ae(outputs: dict[str, np.ndarray], x: np.ndarray, flags) -> pd.DataFrame:
    ref = flags(outputs["pytorch_cpu"], x)
    rows = {k: {"flag_change_%": round(100 * float((flags(v, x) != ref).mean()), 4),
                "max_abs_diff_recon": round(float(np.abs(v - outputs["pytorch_cpu"]).max()), 5)}
            for k, v in outputs.items() if k != "pytorch_cpu"}
    return pd.DataFrame(rows).T


def calib_windows(wide: pd.DataFrame, before: pd.Timestamp, n: int, seed: int = 0):
    """시험 이전(학습+검증) 기간 창에서 n 개 무작위 — GRU 용, 오토인코더 용."""
    import torch

    from tsinfer.models import anomaly as an
    from tsinfer.models import forecast as fc

    rng = np.random.default_rng(seed)
    pre = wide.loc[wide.index < before]
    _, spec = fc.load()
    xg, _, _ = fc.make_windows(pre, spec, with_target=False)
    xg = xg[rng.choice(len(xg), min(n, len(xg)), replace=False)] if len(xg) else xg
    xa = None
    if an.MODEL_PATH.exists():
        ckpt = torch.load(an.MODEL_PATH, map_location="cpu", weights_only=False)
        spaces = [sp for sp in pre.columns.get_level_values(0).unique()
                  if all((sp, m) in pre and pre[(sp, m)].notna().any() for m in an.METRICS)]
        xa = np.concatenate([an.ae_windows(pre, sp, ckpt["stats"], ckpt["L"])[0] for sp in spaces])
        xa = xa[rng.choice(len(xa), min(n, len(xa)), replace=False)] if len(xa) else xa
    return xg, xa


# ---------------------------------------------------------------- 실행

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-int8", action="store_true")
    ap.add_argument("--skip-memory", action="store_true")
    ap.add_argument("--calib-n", type=int, default=512, help="보정에 쓸 시험 이전 창 수")
    ap.add_argument("--check", type=int, default=2048, help="변환 오차를 비교할 시험 창 수")
    ap.add_argument("--repeat", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--chronos-model", default="amazon/chronos-bolt-small")
    args = ap.parse_args()

    t_start = time.perf_counter()
    cfg = load_yaml("default.yaml")
    threads = cfg.get("model", {}).get("threads", 3)
    batches = [1, 4]
    WORKDIR.mkdir(parents=True, exist_ok=True)
    lines_mem, lines_int8 = [], []

    # (b) 먼저: 데이터를 올리기 전 깨끗한 프로세스에서 잰다
    if not args.skip_memory:
        state_path, state_note = _state_model_path(cfg)
        gc.collect()
        avail0, rss0 = mem_available_mb(), rss_mb()
        steps, keep = load_steps(memory_steps(cfg, args.chronos_model, state_path))
        avail1 = mem_available_mb()
        total = float(steps["rss_increase_mb"].sum(skipna=True))
        comp, combos = default_combos(steps)
        combo = suggest_combos(comp, combos, avail0)
        del keep
        gc.collect()
        lines_mem = [
            "## 2. 모델 동시 적재 메모리",
            "",
            f"- 순서대로 한 프로세스에 올림 (각 단계 = 불러오기 + 추론 1회). 시작 RSS {rss0:.0f}MB (파이썬·numpy·pandas), "
            f"시작 때 보드 여유(MemAvailable) {avail0:.0f}MB → 끝 {avail1:.0f}MB",
            f"- 상태 분류기: {state_note}. ONNX 행은 edge.py 가 만든 models/*.onnx 가 있을 때만",
            "- RSS 는 CPU 쪽 메모리. ONNX·torch 를 둘 다 올린 것은 비교용이며 실제 서비스는 둘 중 하나만 쓴다",
            "- 보드 여유는 공용 장비라 다른 팀원 작업에 따라 바뀐다 (측정 시점 값)",
            "",
            steps.to_markdown(index=False),
            "",
            f"- RSS 증가 합계 (모든 단계): {total:.0f}MB",
            "",
            "### 로컬 LLM 과 함께 올리는 조합 (제안)",
            "",
            f"- LLM 크기는 **가정**이다 (실측 아님): {', '.join(LLM_ASSUMED_MB)}. 4bit 가중치 + KV 캐시·런타임 대략값",
            f"- left_mb = 시작 때 보드 여유 − 모델 RSS 합 − LLM. 판정: ≥{HEADROOM_OK_MB}MB 여유, "
            f"≥{HEADROOM_TIGHT_MB}MB 빠듯, 그 아래 부족 (스왑에 들어가면 지연이 크게 튄다)",
            "",
            combo.to_markdown(index=False),
            "",
        ]

    # (a) INT8
    if not args.skip_int8:
        from tsinfer.data.preprocess import clean, load_raw, to_wide
        from tsinfer.data.split import split_by_time

        wide = to_wide(clean(load_raw(metrics=cfg["metrics"]), cfg)[0], cfg)
        _, _, test = split_by_time(wide, cfg)
        trtexec = edge.find_trtexec()
        try:
            import tensorrt as trt
            trt_note = f"TensorRT 파이썬 {trt.__version__}"
        except ImportError:
            trt_note = "TensorRT 파이썬 없음"
        lat_all, sections, tries = [], [], []
        with edge.Tegrastats() as tg:
            try:
                xg_cal, xa_cal = calib_windows(wide, test.index.min(), args.calib_n)
                model, x, to_ppm = edge.gru_target(wide, test.index)
                lat, outs, tr = int8_for_model("forecast_gru", model, x, xg_cal, batches, min(args.check, len(x)),
                                               trtexec, args.repeat, args.warmup, threads, WORKDIR)
                lat_all += [dict(r, model="forecast_gru") for r in lat]
                tries += tr
                sections += ["### 예측 GRU: PyTorch CPU 대비 (CO₂ ppm, 환기 판정 = 60분 안 예측 ≥ "
                             f"{cfg['thresholds']['co2_ventilate_ppm']}ppm)", "",
                             accuracy_gru(outs, to_ppm, cfg["thresholds"]["co2_ventilate_ppm"]).to_markdown(), ""]
                if xa_cal is not None:
                    model, x, flags = edge.ae_target(wide, test.index)
                    n = min(args.check * 4, len(x))
                    lat, outs, tr = int8_for_model("anomaly_ae", model, x, xa_cal, batches, n, trtexec,
                                                   args.repeat, args.warmup, threads, WORKDIR)
                    lat_all += [dict(r, model="anomaly_ae") for r in lat]
                    tries += tr
                    sections += [f"### 오토인코더: PyTorch CPU 대비 이상 판정 변화율 (%, {n}개 창)", "",
                                 accuracy_ae(outs, x, flags).to_markdown(), ""]
                else:
                    sections += ["### 오토인코더", "", "models/anomaly_ae.pt 가 없어 건너뜀", ""]
            except FileNotFoundError as e:
                sections += [f"모델 파일이 없어 건너뜀 ({e}) — 과제 3·4 먼저 실행", ""]
        hw = tg.summary()
        table = (pd.DataFrame(lat_all).pivot_table(index=["model", "method"], columns="batch",
                                                   values=["mean_ms", "p95_ms"]).round(3).to_markdown()
                 if lat_all else "(측정된 지연 없음)")
        lines_int8 = [
            "## 1. INT8 변환",
            "",
            f"- {trt_note}, trtexec {'있음' if trtexec else '없음'}. 보정 데이터: 시험 이전(학습+검증) 기간 창 "
            f"{args.calib_n}개 무작위 (시험 기간은 보정에 쓰지 않음)",
            "- 1순위 TensorRT 파이썬 엔트로피 보정(IInt8EntropyCalibrator2), 실패하면 onnxruntime 정적 양자화 Q/DQ ONNX → "
            "trtexec --int8. 두 경우 모두 --fp16 을 함께 켜서 INT8 이 안 되는 층은 FP16 으로 둔다",
            "- TensorRT 지연은 trtexec Latency (입출력 복사 포함, 워밍업 500ms + 200회). onnxruntime 은 "
            f"반복 {args.repeat}회·워밍업 {args.warmup}회, CPU 스레드 {threads}",
            "",
            "### 시도 기록",
            "",
            pd.DataFrame(tries).to_markdown(index=False) if tries else "(없음)",
            "",
            "### 지연 (ms)",
            "",
            table,
            "",
            *sections,
            "### 측정 중 장비 기록 (tegrastats)",
            "",
            pd.Series(hw).to_frame("value").to_markdown(),
            "",
        ]
    shutil.rmtree(WORKDIR, ignore_errors=True)

    lines = [
        "# 과제 7 심화. INT8 · 동시 적재 메모리",
        "",
        "## 측정 조건",
        "- 장비: Jetson Orin Nano Super 8GB, JetPack 7.2.1, 전력 모드 MAXN_SUPER(변경 안 함), 클럭 고정 안 함, "
        "gpu-run.sh 락 안에서 실행",
        f"- 배치 {batches}, 변환 비교 창 GRU {args.check}개·오토인코더 {args.check * 4}개 (시험 기간)",
        f"- 실행 시간 {time.perf_counter() - t_start:.0f}s",
        "",
        *lines_int8,
        *lines_mem,
        "## 해석 (직접 작성)",
        "- INT8 이 FP16 보다 빨라졌나, 정확도(ppm·판정 변화)는 얼마나 잃었나, GRU 에서 INT8 이 안 된 층과 이유:",
        "- 이 모델 크기에서 INT8 이 의미 있는가 (이미 지연이 0.01ms 수준이면 이득이 거의 없음):",
        "- 로컬 LLM 과 함께 올릴 조합 결론 (어느 모델을 빼거나 ONNX 로 바꿀지):",
        "",
    ]
    out = ROOT / "reports" / "07b_edge_deep.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
