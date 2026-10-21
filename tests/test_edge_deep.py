import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from tsinfer.models import anomaly as an  # noqa: E402
from tsinfer.models import edge_deep as ed  # noqa: E402

MEMINFO = """MemTotal:        7999999 kB
MemFree:          100000 kB
MemAvailable:    4096000 kB
"""


def test_mem_available_and_steps():
    assert ed.mem_available_mb(MEMINFO) == 4000
    assert np.isnan(ed.mem_available_mb("nothing"))

    def bad():
        raise FileNotFoundError("models/x.pt")

    steps, keep = ed.load_steps([("big", lambda: np.ones(20_000_000)), ("skip", lambda: None), ("bad", bad)])
    s = steps.set_index("step")
    assert s.loc["big", "rss_increase_mb"] > 100            # 160MB 배열
    assert s.loc["skip", "note"] == "건너뜀" and s.loc["bad", "note"].startswith("실패")
    assert np.isnan(s.loc["bad", "rss_increase_mb"]) and len(keep) == 2


def test_suggest_combos():
    comp = {"a": 300.0, "b": 200.0, "c": np.nan}
    t = ed.suggest_combos(comp, {"ab": ["a", "b"], "ac": ["a", "c"]}, available_mb=3000,
                          llms={"LLM 3B": 2000})
    r = t.set_index(["combo", "llm"])
    assert r.loc[("ab", "LLM 없음"), "left_mb"] == 2500 and r.loc[("ab", "LLM 없음"), "verdict"] == "여유"
    assert r.loc[("ab", "LLM 3B"), "left_mb"] == 500 and r.loc[("ab", "LLM 3B"), "verdict"] == "빠듯"
    assert "c" in r.loc[("ac", "LLM 3B"), "note"]
    steps = pd.DataFrame({"step": ["torch import", "GRU (torch)", "Chronos-Bolt (x)"], "rss_increase_mb": [200, 5, 300]})
    comp, combos = ed.default_combos(steps)
    assert "Chronos-Bolt (x)" in combos["서비스 torch + Chronos-Bolt"]


def test_int8_layer_share():
    info = {"Layers": [{"Name": "fc", "Outputs": [{"Format/Datatype": "Row major Int8 format"}]},
                       {"Name": "gru", "Outputs": [{"Format/Datatype": "Half"}]}, "plain_name"]}
    assert ed.int8_layer_share(info) == (1, 3)


def test_int8_without_tensorrt_uses_qdq_path(tmp_path):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = an.build_ae(12 * an.CH).eval()
    x = rng.normal(size=(256, 12 * an.CH)).astype("float32")
    calib = rng.normal(size=(128, 12 * an.CH)).astype("float32")
    lat, outs, tries = ed.int8_for_model("ae", model, x, calib, [1, 4], 64, None, repeat=3, warmup=1,
                                         threads=1, workdir=tmp_path)
    paths = {t["path"]: t["result"] for t in tries}
    assert paths["onnxruntime 정적 양자화 (Q/DQ ONNX)"] == "성공"
    assert "trtexec 없음" in paths["TensorRT FP16 (기준)"]
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        assert paths["TensorRT 파이썬 IInt8EntropyCalibrator2"].startswith("실패")
    assert "onnxruntime_int8_qdq" in outs and outs["onnxruntime_int8_qdq"].shape == outs["pytorch_cpu"].shape
    assert {r["method"] for r in lat} == {"onnxruntime_cpu", "onnxruntime_int8_qdq"}

    def flags(out, inp):
        return ((out - inp[:len(out)]) ** 2).mean(axis=1) > 1.0
    acc = ed.accuracy_ae(outs, x, flags)
    assert acc.loc["onnxruntime_int8_qdq", "max_abs_diff_recon"] < 1.0      # INT8 이라 약간 다르지만 크게는 안 틀림
    g = ed.accuracy_gru({"pytorch_cpu": np.array([[990.0, 995]]), "int8": np.array([[990.5, 1001]])},
                        lambda v: v, 1000)
    assert g.loc["int8", "max_abs_diff_ppm"] == 6 and g.loc["int8", "vent_flag_change_%"] == 100
