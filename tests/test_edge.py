import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from tsinfer.models import anomaly as an  # noqa: E402
from tsinfer.models import edge  # noqa: E402
from tsinfer.models import forecast as fc  # noqa: E402

TRT_LOG = """[I] === Performance summary ===
[I] Throughput: 4000 qps
[I] Latency: min = 0.20 ms, max = 0.90 ms, mean = 0.25 ms, median = 0.24 ms, percentile(95%) = 0.31 ms
[I] GPU Compute Time: min = 0.1 ms, max = 0.5 ms, mean = 0.12 ms, median = 0.11 ms, percentile(95%) = 0.15 ms
"""


def test_parse_trtexec(tmp_path):
    assert edge.parse_trtexec_latency(TRT_LOG) == {"mean_ms": 0.25, "p95_ms": 0.31}
    assert edge.parse_trtexec_latency("nothing") is None
    p = tmp_path / "o.json"
    p.write_text(json.dumps([{"name": "y", "dimensions": "2x3", "values": [1, 2, 3, 4, 5, 6]}]))
    assert edge.parse_trtexec_output(p).shape == (2, 3)


@pytest.mark.parametrize("kind", ["gru", "ae"])
def test_onnx_matches_torch(tmp_path, kind):
    rng = np.random.default_rng(0)
    if kind == "gru":
        spec = fc.FeatureSpec(spaces=["a", "b"], outdoor_space=None, lookback=24, horizons=list(range(1, 13)),
                              step_minutes=5, timezone="Asia/Seoul")
        model = fc.build_model(spec, {"hidden": 16, "layers": 1, "dropout": 0.0}).eval()
        x = rng.normal(size=(64, 24, spec.n_features)).astype("float32")
    else:
        model = an.build_ae(12 * an.CH).eval()
        x = rng.normal(size=(64, 12 * an.CH)).astype("float32")
    path = edge.export_onnx(model, x, tmp_path / f"{kind}.onnx")
    sess = edge.ort_session(path, threads=1)
    ref = edge.torch_run(model, x)
    out = edge.ort_run(sess, x)
    assert out.shape == ref.shape and np.abs(out - ref).max() < 1e-4
    assert edge.ort_run(sess, x[:4]).shape[0] == 4          # 배치 크기가 바뀌어도 동작
    t = edge.time_fn(lambda: sess.run(None, {"x": x[:1]}), repeat=5, warmup=1)
    assert t["mean_ms"] > 0 and t["p95_ms"] >= 0
