import numpy as np
import pandas as pd

from tsinfer.data.diagnose import build, device_intervals, long_gaps, odd_days, rise_events

CFG = {
    "spaces": ["lab"],
    "metrics": ["co2"],
    "timezone": "Asia/Seoul",
    "preprocess": {"resample": "5min", "valid_range": {"co2": [350, 5000]}, "short_gap_max_minutes": 10},
}


def _office(days=21, start="2026-09-01"):
    """평일 09~12시(KST)에 사람이 있어 CO₂ 가 오르고, 이후 감쇠. 한 날은 하루 종일 높다."""
    ts = pd.date_range(start, periods=days * 24 * 60, freq="1min", tz="Asia/Seoul")
    co2 = np.full(len(ts), 450.0)
    level = 450.0
    for i, t in enumerate(ts):
        occupied = t.dayofweek < 5 and 9 <= t.hour < 12
        target = 1100.0 if occupied else 450.0
        level += (target - level) * (0.03 if occupied else 0.02)
        co2[i] = level
    odd = (ts.date == pd.Timestamp("2026-09-10").date())
    co2[odd] = 1500.0
    df = pd.DataFrame({"ts": ts.tz_convert("UTC"), "space": "lab", "device": "a", "metric": "co2", "value": co2})
    slow = df.iloc[::10].assign(device="b")   # 10분 주기 센서
    return pd.concat([df, slow], ignore_index=True)


def test_device_intervals_detects_report_period():
    iv = device_intervals(_office(days=1))
    assert iv.loc[("lab", "a", "co2"), "median_interval_min"] == 1.0
    assert iv.loc[("lab", "b", "co2"), "median_interval_min"] == 10.0


def test_long_gaps_found():
    idx = pd.date_range("2026-09-01", periods=100, freq="5min", tz="UTC")
    wide = pd.DataFrame({("lab", "co2"): 500.0}, index=idx)
    wide.iloc[20:44] = np.nan      # 2시간
    g = long_gaps(wide)
    assert len(g) == 1 and g.loc[0, "hours"] == 2.0


def test_rise_events_and_odd_day():
    raw = _office()
    lines, profiles = build(raw, CFG)
    from tsinfer.data.preprocess import clean, to_wide
    wide = to_wide(clean(raw, CFG)[0], CFG)
    ev = rise_events(wide[("lab", "co2")], "Asia/Seoul")
    assert len(ev) >= 10
    assert (ev["start_local"].dt.hour == 9).mean() > 0.8
    assert ev["to_peak_min"].median() > 60 and ev["half_decay_min"].notna().mean() > 0.8
    odd = odd_days(wide[("lab", "co2")], "Asia/Seoul")
    assert str(odd.loc[0, "date"]) == "2026-09-10"
    prof = profiles["lab"]
    assert prof.loc[11, "평일"] > 900 and prof.loc[11, "주말"] < 500
    assert any("데이터 진단서" in line for line in lines)
