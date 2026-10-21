import numpy as np
import pandas as pd

from tsinfer.models import pretrained as pt
from tsinfer.models import pretrained_deep as pd_


def _wide():
    idx = pd.date_range("2026-10-06", periods=288 * 3, freq="5min", tz="UTC")
    v = 500 + np.arange(len(idx), dtype=float)          # 칸마다 +1
    w = pd.DataFrame({("lab", "co2"): v, ("lab", "temperature"): 22.0, ("lab", "humidity"): 40.0,
                      ("out", "temperature"): np.nan, ("out", "humidity"): 60.0}, index=idx)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["space", "metric"])
    return w


def _col():
    """가짜 예측: GRU 는 +4 치우침, Chronos 는 -4 치우침 → 평균이 정확."""
    w = _wide()
    pts = pt.eval_points(w.index, 60, 1000, 60)

    def chronos(ctx, H):
        last = ctx[:, -1][:, None] + np.arange(1, H + 1)[None, :] - 4
        return np.stack([last - 2, last, last + 2], axis=2)

    y = {m: pt.truth(w[("lab", m)], pts, list(range(1, 13)), 5) for m in pt.TARGETS}
    gru = {("lab", m): y[m] + 4 for m in pt.TARGETS}
    gru[("lab", "temperature")] = y["temperature"] + 4.0
    return w, pts, pd_.collect(w, pts, ["lab"], chronos, 24, 12, gru)


def test_weights_and_ensemble():
    w, pts, col = _col()
    weights = pd_.fit_weights(col, 5, [5, 60])
    assert weights[("co2", 60)] == 0.5
    res = pd_.ensemble_rows(col, 5, [5, 60], weights).set_index(["metric", "horizon_min"])
    r = res.loc[("co2", 60)]
    assert r["gru"] == 4 and r["chronos"] == 4 and r["mean_0.5"] == 0 and r["weighted"] == 0 and r["persistence"] == 12
    wins = pd_.ensemble_wins(res.reset_index())
    assert set(wins["method"]) == {"mean_0.5", "weighted"} and (wins["beats_both"] > 0).all()


def test_no_gru_means_no_ensemble_rows():
    w, pts, col = _col()
    for d in col.values():
        d.pop("gru")
    assert pd_.fit_weights(col, 5, [5]) == {}
    res = pd_.ensemble_rows(col, 5, [5], {}, extra={"chronos2": {("lab", "co2"): col[("lab", "co2")]["chronos"]}})
    assert "mean_0.5" not in res and "chronos2" in res and pd_.ensemble_wins(res).empty


def test_chronos2_load_failure_is_reported():
    def boom(name, device):
        raise OSError("no network")

    pipe, status = pd_.try_load_chronos2("amazon/chronos-2", "cpu", loader=boom)
    assert pipe is None and "불러오기 실패" in status

    def missing(name, device):
        raise ImportError("no chronos")

    assert pd_.try_load_chronos2("x", "cpu", loader=missing)[1].startswith("설치 안 됨")


def test_covariate_inputs_shapes():
    w = _wide()
    pts = w.index[[3, 400]]
    sched = {"weekly": {"lab": [{"days": ["mon", "tue", "wed", "thu", "fri"], "start": "09:00", "end": "18:00"}]}}
    items = pd_.covariate_inputs(w, "lab", "co2", pts, 24, 12, sched, "out", "Asia/Seoul", 5)
    assert len(items) == 2
    for it in items:
        assert it["target"].shape == (24,)
        assert set(it["past_covariates"]) == {"own_temperature", "own_humidity", "out_temperature", "out_humidity",
                                              "schedule"}
        assert all(v.shape == (24,) and not np.isnan(v).any() for v in it["past_covariates"].values())
        assert it["future_covariates"]["schedule"].shape == (12,)
    assert np.isnan(items[0]["target"][:20]).all()            # 데이터 시작 전은 목표 NaN
    assert set(pd_.covariate_inputs(w, "lab", "co2", pts, 24, 12, use_covariates=False)[0]) == {"target"}


def test_chronos2_predict_with_fake_pipeline():
    import torch

    class Fake:
        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            q = [torch.full((1, prediction_length, len(quantile_levels)), float(it["target"][-1])) for it in inputs]
            return q, None

    items = [{"target": np.array([1.0, 2.0, float(i)])} for i in range(70)]
    out = pd_.chronos2_predict(Fake(), items, 12)
    assert out.shape == (70, 12, 3) and out[5, 0, 1] == 5
