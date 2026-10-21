from datetime import timezone

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tsinfer.models import forecast as fc  # noqa: E402
from tsinfer.service.predictors import GRUForecaster  # noqa: E402

CFG = {
    "spaces": ["lab", "meet", "outside"],
    "metrics": ["co2", "temperature", "humidity"],
    "timezone": "Asia/Seoul",
    "split": {"train_end": "2026-09-15", "valid_end": "2026-09-19"},
    "forecast": {"horizon_minutes": 60, "step_minutes": 5, "report_horizons": [5, 15, 30, 60]},
    "model": {"lookback_minutes": 60, "hidden": 32, "layers": 1, "dropout": 0.0, "epochs": 8,
              "batch_size": 128, "lr": 0.005, "patience": 3, "threads": 2, "outdoor_space": "outside", "seed": 0},
}


def _wide(days=22, start="2026-09-01"):
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC")
    local = idx.tz_convert("Asia/Seoul")
    hour = local.hour + local.minute / 60
    occ = (local.dayofweek < 5) & (hour >= 9) & (hour < 12)
    data = {}
    rng = np.random.default_rng(0)
    for sp, amp in (("lab", 600), ("meet", 400)):
        co2 = np.empty(len(idx))
        lvl = 450.0
        for i, o in enumerate(occ):
            lvl += ((450 + amp) - lvl) * 0.08 if o else (450 - lvl) * 0.06
            co2[i] = lvl
        data[(sp, "co2")] = co2 + rng.normal(0, 3, len(idx))
        data[(sp, "temperature")] = 22 + 2 * occ + np.sin(2 * np.pi * hour / 24)
        data[(sp, "humidity")] = 40 + 5 * occ
    data[("outside", "co2")] = np.nan
    data[("outside", "temperature")] = 15 + 5 * np.sin(2 * np.pi * hour / 24)
    data[("outside", "humidity")] = 60.0
    w = pd.DataFrame(data, index=idx)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["space", "metric"])
    return w


def test_windows_shapes_and_decode_zero_is_persistence():
    w = _wide(days=6, start="2026-09-12")
    spec = fc.fit_spec(w, w.index[:1000], CFG)
    assert spec.spaces == ["lab", "meet"] and spec.lookback == 12 and len(spec.horizons) == 12
    x, y, info = fc.make_windows(w, spec)
    assert x.shape[1:] == (12, spec.n_features) and y.shape[1] == 24 and len(x) == len(info)
    out = fc.decode(np.zeros((len(x), 24), "float32"), info, spec)
    assert np.allclose(out["co2"][:, -1], info["cur_co2"])
    # 자기 공간 CO₂ 결측이 창에 있으면 버린다
    w2 = w.copy()
    w2.iloc[100, w2.columns.get_loc(("lab", "co2"))] = np.nan
    x2, _, _ = fc.make_windows(w2, spec)
    assert len(x2) == len(x) - 12 - 12   # 그 칸을 포함하는 창 12개 + 그 칸을 목표로 갖는 창 12개


def test_gru_beats_persistence_and_service_predictor(tmp_path):
    w = _wide()
    model, spec, res, history, _ = fc.run(CFG, w, log=lambda *_: None)
    r = res[(res["metric"] == "co2") & (res["horizon_min"] == 60)].set_index("space")
    assert (r["gru"] < r["persistence"]).all()
    assert len(history) >= 1

    path = tmp_path / "m.pt"
    fc.save(model, spec, CFG["model"], path)
    f = GRUForecaster(path)
    now = pd.Timestamp("2026-09-21 01:00", tz="UTC")   # 월요일 10시 KST, 사용 중
    p = f.predict(w[w.index <= now], "lab")
    assert set(p) == {"co2", "temperature"} and set(p["co2"]) == set(range(5, 61, 5))
    assert p["co2"][60] > w.loc[now, ("lab", "co2")]
    assert f.predict(w[w.index <= now], "outside") == {}
    # 10분 주기 센서처럼 끝 칸이 비어도 예측한다 (직전 값으로 최대 2칸), 3칸 넘게 비면 끊김으로 본다
    cut = w[w.index <= now].copy()
    cut.iloc[-1, cut.columns.get_loc(("lab", "co2"))] = np.nan
    assert set(f.predict(cut, "lab")) == {"co2", "temperature"}
    cut.iloc[-3:, cut.columns.get_loc(("lab", "co2"))] = np.nan
    assert f.predict(cut, "lab") == {}


def test_holidays_schedule_and_alpha():
    import copy

    cfg = copy.deepcopy(CFG)
    cfg["holidays"] = ["2026-09-16"]                    # 검증 기간 수요일을 공휴일로
    cfg["model"]["epochs"] = 2
    sched = {"weekly": {sp: [{"days": ["mon", "tue", "wed", "thu", "fri"], "start": "09:00", "end": "12:00"}]
                        for sp in ("lab", "meet")}}
    w = _wide()
    model, spec, res, _, _ = fc.run(cfg, w, log=lambda *_: None, schedule=sched)
    assert spec.n_features == fc.FeatureSpec(spaces=spec.spaces, outdoor_space="outside", lookback=12,
                                             horizons=spec.horizons, step_minutes=5,
                                             timezone="Asia/Seoul").n_features + len(fc.SCHEDULE_OFFSETS)
    assert set(spec.alpha) == {"co2", "temperature"} and all(0 <= a <= 1.2 for a in spec.alpha["co2"])
    hol = pd.Timestamp("2026-09-16 01:00", tz="UTC")     # 수요일 10시 KST
    assert not fc.is_workday(pd.DatetimeIndex([hol]), spec)[0]
    f = fc.space_features(w, "lab", spec)
    assert f.loc[hol, "sched_0"] == 0 and f.loc[pd.Timestamp("2026-09-15 01:00", tz="UTC"), "sched_0"] == 1
    assert f.loc[pd.Timestamp("2026-09-14 23:30", tz="UTC"), "sched_60"] == 1   # 08:30 KST → 1시간 뒤 수업
