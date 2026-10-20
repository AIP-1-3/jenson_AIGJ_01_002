import numpy as np
import pandas as pd
import pytest

from tsinfer.models import anomaly as an

CFG = {
    "spaces": ["lab", "meet", "outside"],
    "metrics": ["co2", "temperature", "humidity"],
    "split": {"train_end": "2026-09-15", "valid_end": "2026-09-19"},
    "model": {"threads": 2},
}


def _wide(days=24, start="2026-09-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC")
    local = idx.tz_convert("Asia/Seoul")
    hour = local.hour + local.minute / 60
    occ = ((local.dayofweek < 5) & (hour >= 9) & (hour < 12)).astype(float)
    data = {}
    for sp, amp in (("lab", 600), ("meet", 400)):
        lvl, co2 = 450.0, np.empty(len(idx))
        for i, o in enumerate(occ):
            lvl += ((450 + amp) - lvl) * 0.08 if o else (450 - lvl) * 0.06
            co2[i] = lvl
        data[(sp, "co2")] = co2 + rng.normal(0, 3, len(idx))
        data[(sp, "temperature")] = 22 + 2 * occ + np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.05, len(idx))
        data[(sp, "humidity")] = 40 + 5 * occ + rng.normal(0, 0.3, len(idx))
    data[("outside", "co2")] = np.nan
    data[("outside", "temperature")] = 15.0
    data[("outside", "humidity")] = 60.0
    w = pd.DataFrame(data, index=idx)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["space", "metric"])
    return w


def test_inject_places_non_overlapping_faults():
    w = _wide(days=8)
    inj, faults = an.inject(w, ["lab", "meet"], per_kind=3, seed=1)
    assert len(faults) == 15 and {f.kind for f in faults} == set(an.KINDS)
    for sp in ("lab", "meet"):
        fs = sorted((f.start, f.end) for f in faults if f.space == sp)
        assert all(a[1] <= b[0] for a, b in zip(fs, fs[1:]))
    d = next(f for f in faults if f.kind == "dropout")
    assert inj.loc[d.start, (d.space, "co2")] != inj.loc[d.start, (d.space, "co2")]   # NaN


def test_rules_catch_spike_stuck_gap_without_false_alarms_on_clean():
    w = _wide(days=8)
    inj, faults = an.inject(w, ["lab", "meet"], per_kind=3, seed=2)
    days = 8.0
    res = an.evaluate(an.rule_flags(inj, ["lab", "meet"]), faults, days)
    for kind in ("spike", "stuck", "dropout"):
        assert res.loc[kind, "detection_rate"] == 100.0
    assert an.evaluate(an.rule_flags(w, ["lab", "meet"]), [], days).loc["전체", "false_alarms_per_day"] == 0


def test_full_run_combined_beats_rules():
    pytest.importorskip("torch")
    ae, spaces, results, clean_fa, curve, faults, _ = an.run(CFG, _wide(), per_kind=6, epochs=8)
    assert spaces == ["lab", "meet"] and np.isfinite(ae.threshold)
    comb, rules = results["combined"].loc["전체"], results["rules"].loc["전체"]
    assert comb["detection_rate"] >= rules["detection_rate"]
    assert results["ae"].loc["noise", "detection_rate"] > 50
    assert len(curve) == 8
