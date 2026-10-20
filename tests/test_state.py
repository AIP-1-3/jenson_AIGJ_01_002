import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from tsinfer.models import state as st  # noqa: E402

CFG = {
    "spaces": ["lab", "meet", "outside"],
    "metrics": ["co2", "temperature", "humidity"],
    "timezone": "Asia/Seoul",
    "split": {"train_end": "2026-09-15", "valid_end": "2026-09-19"},
    "forecast": {"horizon_minutes": 60, "step_minutes": 5, "report_horizons": [5, 15, 30, 60]},
    "thresholds": {"co2_ventilate_ppm": 1000, "co2_critical_ppm": 1500, "ventilation_alert_within_minutes": 30},
    "model": {"outdoor_space": "outside", "seed": 0},
}
SCHED = {"weekly": {sp: [{"days": ["mon", "tue", "wed", "thu", "fri"], "start": "09:00", "end": "12:00"}]
                    for sp in ("lab", "meet")},
         "exceptions": [{"space": "lab", "date": "2026-09-22", "start": "00:00", "end": "24:00", "occupied": False}]}


def _wide(days=24, start="2026-09-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC")
    local = idx.tz_convert("Asia/Seoul")
    hour = local.hour + local.minute / 60
    occ = (local.dayofweek < 5) & (hour >= 9) & (hour < 12)
    data = {}
    for sp, amp in (("lab", 900), ("meet", 450)):
        lvl, co2 = 450.0, np.empty(len(idx))
        for i, o in enumerate(occ):
            people = amp * (0.7 + 0.6 * rng.random()) if o else 0     # 날마다 인원이 다르다
            lvl += ((450 + people) - lvl) * 0.06 if o else (450 - lvl) * 0.06
            co2[i] = lvl
        data[(sp, "co2")] = co2 + rng.normal(0, 5, len(idx))
        data[(sp, "temperature")] = 22 + 1.5 * occ + rng.normal(0, 0.1, len(idx))
        data[(sp, "humidity")] = 40 + 4 * occ + rng.normal(0, 0.5, len(idx))
    data[("outside", "co2")] = np.nan
    data[("outside", "temperature")] = 15 + 5 * np.sin(2 * np.pi * hour / 24)
    data[("outside", "humidity")] = 60.0
    w = pd.DataFrame(data, index=idx)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["space", "metric"])
    return w


def test_labels():
    idx = pd.date_range("2026-09-21 23:00", periods=6, freq="1h", tz="UTC")   # KST 월요일 08~13시
    occ = st.occupancy_labels(idx, ["lab", "meet"], SCHED,
                              pd.DataFrame({"space": ["meet"], "start": ["2026-09-22 11:00"],
                                            "end": ["2026-09-22 12:00"], "occupied": [0]}))
    assert occ["lab"].sum() == 0                     # 예외: 이날 실습실 비어 있음
    assert occ["meet"].tolist() == [0, 1, 1, 0, 0, 0]  # 9·10시 사용, 11시는 관찰 기록으로 비어 있음
    co2 = pd.Series([900, 950, 990, 1010, 1100, 980, 970, 960.0])
    y = st.ventilation_labels(co2, 1000, 2)
    assert y.tolist()[:3] == [0.0, 1.0, 1.0] and np.isnan(y.iloc[3]) and np.isnan(y.iloc[4])


def test_classifiers_beat_rules_and_vent_recall():
    res, fitted, thr, errors, feats, df, info = st.run(CFG, _wide(), SCHED)
    occ, vent = res["occupied"], res["ventilation"]
    assert max(occ.loc["logreg", "macro_f1"], occ.loc["hgb", "macro_f1"]) > occ.loc["rule", "macro_f1"]
    assert vent.loc["hgb", "recall"] >= 0.8
    assert "y_occ" not in feats and "y_vent" not in feats
    assert {"occupied_hgb", "ventilation_logreg"} <= set(fitted)
