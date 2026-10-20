import numpy as np
import pandas as pd

from tsinfer.data.baselines import evaluate
from tsinfer.data.preprocess import clean, to_wide
from tsinfer.data.split import fit_scaler, split_by_time

CFG = {
    "spaces": ["lab"],
    "metrics": ["co2"],
    "preprocess": {"resample": "5min", "valid_range": {"co2": [350, 5000]}, "short_gap_max_minutes": 10},
    "split": {"train_end": "2026-08-03", "valid_end": "2026-08-04"},
}


def _long(n_min=60 * 24 * 4, start="2026-08-01"):
    ts = pd.date_range(start, periods=n_min, freq="1min", tz="UTC")
    rows = []
    for dev, off in (("a", 0.0), ("b", 10.0)):
        v = 600 + 200 * np.sin(np.arange(n_min) / (60 * 24) * 2 * np.pi) + off
        rows.append(pd.DataFrame({"ts": ts, "space": "lab", "device": dev, "metric": "co2", "value": v}))
    return pd.concat(rows, ignore_index=True)


def test_clean_removes_impossible_values():
    df = _long(10)
    df.loc[0, "value"] = 0
    out, rep = clean(df, CFG)
    assert rep.removed_out_of_range["co2"] == 1 and len(out) == len(df) - 1


def test_to_wide_averages_devices_and_resamples():
    wide = to_wide(_long(60), CFG)
    assert wide.index.freq == "5min" or (wide.index[1] - wide.index[0]) == pd.Timedelta("5min")
    first = wide[("lab", "co2")].iloc[0]
    assert 600 < first < 620  # 두 장치(0, +10) 평균


def test_short_gap_filled_long_gap_kept():
    df = _long(120)
    t0 = df["ts"].min()
    short = (df["ts"] >= t0 + pd.Timedelta("10min")) & (df["ts"] < t0 + pd.Timedelta("15min"))
    long = (df["ts"] >= t0 + pd.Timedelta("60min")) & (df["ts"] < t0 + pd.Timedelta("90min"))
    wide = to_wide(df[~(short | long)], CFG)[("lab", "co2")]
    assert not np.isnan(wide[t0 + pd.Timedelta("10min")])
    assert np.isnan(wide[t0 + pd.Timedelta("70min")])


def test_split_is_chronological_and_scaler_uses_train_only():
    wide = to_wide(_long(), CFG)
    train, valid, test = split_by_time(wide, CFG)
    assert train.index.max() < valid.index.min() <= valid.index.max() < test.index.min()
    assert np.isclose(fit_scaler(train).loc[("lab", "co2"), "mean"], train[("lab", "co2")].mean())


def test_baselines_persistence_exact_on_constant_and_yesterday_on_daily_cycle():
    wide = to_wide(_long(), CFG)
    _, _, test = split_by_time(wide, CFG)
    res = evaluate(wide, test.index, [5, 60]).set_index(["method", "horizon_min"])["mae"]
    # 정확히 하루 주기 신호 → '어제 같은 시각'이 거의 완벽
    assert res[("yesterday", 60)] < 1e-6
    assert res[("persistence", 60)] > res[("persistence", 5)]
