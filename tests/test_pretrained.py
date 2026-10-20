import numpy as np
import pandas as pd

from tsinfer.models import pretrained as pt


def _wide():
    idx = pd.date_range("2026-10-06", periods=288 * 3, freq="5min", tz="UTC")
    v = 500 + np.arange(len(idx), dtype=float)          # 칸마다 +1
    w = pd.DataFrame({("lab", "co2"): v, ("lab", "temperature"): 22.0}, index=idx)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["space", "metric"])
    return w


def test_eval_points_and_contexts():
    w = _wide()
    pts = pt.eval_points(w.index, 60, 1000, 60)
    assert (pts.minute == 0).all() and pts[-1] <= w.index[-1] - pd.Timedelta("60min")
    assert len(pt.eval_points(w.index, 60, 10, 60)) == 10
    ctx = pt.contexts(w[("lab", "co2")], w.index[[2, 100]], 5)
    assert np.isnan(ctx[0, :2]).all() and ctx[0, -1] == 502 and ctx[1].tolist() == [596, 597, 598, 599, 600]


def test_compare_with_perfect_fake_model():
    w = _wide()
    pts = pt.eval_points(w.index, 60, 1000, 60)

    def fake(ctx, H):   # 추세를 정확히 아는 가짜 모델: 중앙값 = 마지막+h, 구간 ±2
        last = ctx[:, -1][:, None] + np.arange(1, H + 1)[None, :]
        return np.stack([last - 2, last, last + 2], axis=2)

    res, cov = pt.compare(w, pts, ["lab"], fake, 24, 12, 5, [5, 60])
    r = res.set_index(["metric", "horizon_min"])
    assert r.loc[("co2", 60), "chronos"] == 0 and r.loc[("co2", 60), "persistence"] == 12
    assert (cov.loc[cov["metric"] == "co2", "coverage_10_90_%"] == 100).all()
