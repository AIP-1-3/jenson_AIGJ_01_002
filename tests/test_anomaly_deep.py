import numpy as np
import pandas as pd
import pytest

from tsinfer.models import anomaly as an
from tsinfer.models import anomaly_deep as ad

SPACES = ["lab", "meet", "office"]
CFG = {
    "spaces": SPACES + ["outside"],
    "metrics": ["co2", "temperature", "humidity"],
    "timezone": "Asia/Seoul",
    "holidays": [],
    "split": {"train_end": "2026-09-15", "valid_end": "2026-09-19"},
    "forecast": {"horizon_minutes": 60, "step_minutes": 5, "report_horizons": [5, 15, 30, 60]},
    "model": {"lookback_minutes": 60, "hidden": 16, "layers": 1, "dropout": 0.0, "epochs": 2,
              "batch_size": 128, "lr": 0.005, "patience": 3, "threads": 2, "outdoor_space": "outside", "seed": 0},
}


def _wide(days=24, start="2026-09-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 288, freq="5min", tz="UTC")
    local = idx.tz_convert("Asia/Seoul")
    hour = local.hour + local.minute / 60
    occ = ((local.dayofweek < 5) & (hour >= 9) & (hour < 12)).astype(float)
    data = {}
    for sp, amp in (("lab", 600), ("meet", 400), ("office", 300)):
        lvl, co2 = 430.0, np.empty(len(idx))
        for i, o in enumerate(occ):
            lvl += ((430 + amp) - lvl) * 0.08 if o else (430 - lvl) * 0.06
            co2[i] = lvl
        data[(sp, "co2")] = co2 + rng.normal(0, 3, len(idx))
        data[(sp, "temperature")] = 22 + 2 * occ + np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.05, len(idx))
        data[(sp, "humidity")] = 40 + 5 * occ + 3 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.3, len(idx))
    data[("outside", "co2")] = np.nan
    data[("outside", "temperature")] = 15 + 5 * np.sin(2 * np.pi * hour / 24)
    data[("outside", "humidity")] = 60.0
    w = pd.DataFrame(data, index=idx)
    w.columns = pd.MultiIndex.from_tuples(w.columns, names=["space", "metric"])
    return w


def _persistence_gru(w):
    """학습 없이 '현재값 유지'를 내는 GRU (α=0) — 예측 오차 = 지금 값 - h칸 전 값."""
    torch = pytest.importorskip("torch")
    from tsinfer.models import forecast as fc

    torch.manual_seed(0)
    spec = fc.fit_spec(w, w.index[: 288 * 10], CFG)
    spec.alpha = {m: [0.0] * len(spec.horizons) for m in fc.TARGETS}
    return fc.build_model(spec, CFG["model"]).eval(), spec


def _drift(w, sp, metric, start, amp, hours=6):
    w = w.copy()
    t0 = pd.Timestamp(start, tz="UTC")
    idx = w.index[(w.index >= t0) & (w.index < t0 + pd.Timedelta(hours=hours))]
    w.loc[idx, (sp, metric)] += np.linspace(0, amp, len(idx))
    return w, idx


def test_cusum_accumulates_bias_and_ignores_nan():
    assert ad.cusum(np.zeros(50)).max() == 0
    up = ad.cusum(np.full(40, 0.75), k=0.25)
    assert np.allclose(up, np.arange(1, 41) * 0.5)
    down = ad.cusum(np.r_[np.full(10, -1.25), np.full(5, np.nan)], k=0.25)
    assert down[9] == pytest.approx(10.0) and down[-1] == pytest.approx(10.0 - 5 * 0.25)
    assert ad.cusum(np.full(3, 100.0), k=0.0, clip=5.0)[-1] == 15.0


def test_gru_residuals_with_persistence_model_equal_lagged_difference():
    w = _wide(days=3, start="2026-09-12")
    gru = _persistence_gru(w)
    res = ad.gru_residuals(*gru, w)
    r = res[("lab", "co2", 3)]
    expect = w[("lab", "co2")] - w[("lab", "co2")].shift(3)
    ok = r.notna()
    assert ok.sum() > 500 and np.allclose(r[ok], expect[ok], atol=1e-3)
    assert ("lab", "humidity", 1) not in res        # GRU 는 습도를 예측하지 않음


def test_floor_catches_night_co2_drift_but_not_clean_data():
    w = _wide()
    train = w[w.index < "2026-09-15"]
    table = ad.fit_floor(train, SPACES, "Asia/Seoul", [])
    assert table["lab"].shape == (2, 24)
    # 한국 시각 0시(UTC 15시) 시작 → 빈방 새벽 동안 바닥이 올라감
    inj, idx = _drift(w, "lab", "co2", "2026-09-20 15:00", 250)
    clean = ad.floor_scores(w, SPACES, table, "Asia/Seoul", [])
    dirty = ad.floor_scores(inj, SPACES, table, "Asia/Seoul", [])
    test = w.index >= "2026-09-19"
    clean_max = max(clean[sp].loc[test, "co2"].max() for sp in SPACES)
    assert dirty["lab"].loc[idx[-1], "co2"] > clean_max + 50
    assert dirty["lab"]["temperature"].isna().all()   # CO₂ 만


def test_cross_points_to_drifting_space_only():
    w = _wide()
    scale = ad.fit_cross_scale(w[w.index < "2026-09-15"], SPACES)
    inj, idx = _drift(w, "meet", "temperature", "2026-09-20 02:00", 3.0)
    sc = ad.cross_scores(inj, SPACES, scale)
    end = idx[-1]
    assert sc["meet"].loc[end, "temperature"] > 5
    assert np.isnan(sc["lab"].loc[end, "temperature"]) and np.isnan(sc["office"].loc[end, "temperature"])
    clean = ad.cross_scores(w, SPACES, scale)
    assert sc["meet"].loc[end, "temperature"] > clean["meet"]["temperature"].max()


def test_stuck2_scores_frozen_sensor_while_others_move():
    w = _wide()
    move = ad.fit_move_scale(w[w.index < "2026-09-15"], SPACES)
    w2 = w.copy()
    t0 = pd.Timestamp("2026-09-21 23:00", tz="UTC")       # 한국 08시 → 입실로 CO₂·습도가 움직임
    idx = w2.index[(w2.index >= t0) & (w2.index < t0 + pd.Timedelta(hours=3))]
    w2.loc[idx, ("lab", "temperature")] = w2.loc[t0, ("lab", "temperature")]
    sc = ad.stuck2_scores(w2, SPACES, move)["lab"]["temperature"]
    assert np.isnan(sc.loc[idx[5]])                        # 1시간 전에는 보지 않음
    assert sc.loc[idx[-1]] > 3
    assert ad.stuck2_scores(w, SPACES, move)["lab"]["temperature"].isna().all()   # 잡음 있는 원본은 고정 구간 없음


def test_pick_score_threshold_meets_budget():
    idx = pd.date_range("2026-09-01", periods=288 * 4, freq="5min", tz="UTC")
    rng = np.random.default_rng(0)
    scores = {sp: pd.DataFrame({m: rng.normal(size=len(idx)) for m in an.METRICS}, index=idx) for sp in SPACES}
    thr = ad.pick_score_threshold(scores, 4.0, 0.5)
    n = sum(len(an.events((df > thr).any(axis=1))) for df in scores.values())
    assert n / 4.0 <= 0.5
    flags = ad.score_flags(scores, thr, "pe")
    assert list(flags["lab"].columns) == ["pe:co2", "pe:temperature", "pe:humidity"]


def test_fast_scoring_matches_anomaly_evaluate():
    w = _wide(days=8)
    inj, faults = an.inject(w, SPACES, per_kind=3, seed=3)
    flags = an.rule_flags(inj, SPACES)
    ref = an.evaluate(flags, faults, 8.0)
    anyf = ad.any_flags(flags)
    assert ad.fault_hits(anyf, w.index, faults).mean() * 100 == pytest.approx(ref.loc["전체", "detection_rate"], abs=0.1)
    assert ad.false_alarms(anyf, w.index, faults, 8.0) == pytest.approx(ref.loc["전체", "false_alarms_per_day"], abs=0.01)


def test_full_run_selects_combo_on_validation_only():
    pytest.importorskip("torch")
    w = _wide()
    gru = _persistence_gru(w)
    out = ad.run(CFG, w, per_kind=3, epochs=4, gru=gru, log=lambda *a: None)
    assert out["spaces"] == SPACES and set(out["names"]) == {"pe", "cusum", "floor", "cross", "stuck2"}
    best = out["combos"][out["combos"]["combo"] == out["best"]].iloc[0]
    assert best["fa"] <= ad.FA_LIMIT
    base = out["combos"][out["combos"]["combo"] == ()].iloc[0]
    assert best["detection_rate"] >= base["detection_rate"] or base["fa"] > ad.FA_LIMIT
    for m in ("rules", "ae", "pe", "base", "final"):
        assert "전체" in out["results"][m].index
    lines = ad.build_report(out, CFG, 3)
    text = "\n".join(lines)
    assert "## 측정 조건" in text and "## 해석 (직접 작성)" in text and "해당 없음" in text
    # 과제 안내 5종 절: 기존 결합·최종 결합 표, 종류마다 '누가 잡았나' 한 줄, 냉방·환기 정지는 실내·사용 시간만
    assert "## 과제 안내 고장 5종 시험" in text
    assert text.index("## 과제 안내 고장 5종 시험") < text.index("## 설계상 한계")
    for m in ("base", "final"):
        tab = out["t4"][m]["table"]
        assert list(tab.index[:5]) == [f"{k} ({ad.T4_LABELS[k]})" for k in ad.T4_KINDS] and tab.index[-1] == "전체"
        assert ad.METHOD_TITLES[m] in text
    for k in ad.T4_KINDS:
        assert sum(line.startswith(f"- {k} (") for line in lines) == 2
    new = [f for f in out["t4_faults"] if f.kind in ("cooling_off", "vent_off")]
    assert any(f.kind == "cooling_off" for f in new)
    use = ad.usage_mask(w.index, "Asia/Seoul", [])
    for f in new:
        assert f.space in SPACES and use[(w.index >= f.start) & (w.index < f.end)].all()
    assert [f for f in out["t4_faults"] if f.kind in ("spike", "stuck", "drift")] == \
        [f for f in out["faults"] if f.kind in ("spike", "stuck", "drift")]


def test_usage_injections_stay_in_usage_hours_with_right_size():
    w = _wide()
    inj, faults = ad.inject_usage(w, SPACES, per_kind=4, seed=0, tz="Asia/Seoul", holidays=["2026-09-02"])
    kinds = [f.kind for f in faults]
    assert kinds.count("cooling_off") == 4 and kinds.count("vent_off") >= 1
    changed = (inj != w) & inj.notna()
    marked = pd.DataFrame(False, index=w.index, columns=w.columns)
    for f in faults:
        idx = w.index[(w.index >= f.start) & (w.index < f.end)]
        local = idx.tz_convert("Asia/Seoul")
        # 구간 전체가 평일 9~18시, 공휴일 아님, 2~3시간
        assert (local.dayofweek < 5).all() and (local.hour >= 9).all() and (local.hour < 18).all()
        assert not (local.strftime("%Y-%m-%d") == "2026-09-02").any()
        assert 24 <= len(idx) <= 36 and f.end - f.start == len(idx) * pd.Timedelta(minutes=5)
        d = inj.loc[idx, (f.space, f.metric)] - w.loc[idx, (f.space, f.metric)]
        if f.kind == "cooling_off":
            assert f.metric == "temperature"
            assert d.iloc[0] == pytest.approx(0) and 3.0 <= d.iloc[-1] <= 5.0
            assert np.allclose(np.diff(d.to_numpy()), d.iloc[-1] / (len(idx) - 1))      # 선형
        else:
            assert f.metric == "co2"
            c0 = w.loc[idx[0], (f.space, "co2")]
            rise = w.loc[idx, (f.space, "co2")] - c0
            assert rise.iloc[-1] >= ad.VENT_MIN_RISE
            up = rise > 0
            assert np.allclose(inj.loc[idx[up], (f.space, "co2")] - c0, 1.8 * rise[up])
            assert np.allclose(d[~up], 0)
        after = w.index[w.index >= f.end][:1]
        assert not changed.loc[after, (f.space, f.metric)].any()                       # 끝에서 원래 값
        marked.loc[idx, (f.space, f.metric)] = True
    assert not (changed & ~marked).any(axis=None)                                       # 고장 구간 밖은 그대로
    # 같은 공간 고장끼리 겹치지 않고 간격 유지
    for a in faults:
        for b in faults:
            if a is not b and a.space == b.space:
                assert a.end + ad.T4_SEP <= b.start or b.end + ad.T4_SEP <= a.start


def test_task_kinds_reverts_noise_dropout_and_skips_outdoor():
    w = _wide(days=10)
    w[("outside", "co2")] = 420.0 + np.random.default_rng(1).normal(0, 2, len(w))
    spaces = SPACES + ["outside"]
    inj, faults = an.inject(w, spaces, per_kind=2, seed=0)
    t4, t4f = ad.inject_task_kinds(w, inj, faults, spaces, 3, 0, "Asia/Seoul", [], outdoor="outside")
    assert {f.kind for f in t4f} <= set(ad.T4_KINDS)
    assert [f for f in t4f if f.kind in ("spike", "stuck", "drift")] == [f for f in faults if f.kind in ad.T4_KINDS]
    assert all(f.space != "outside" for f in t4f if f.kind in ("cooling_off", "vent_off"))
    for f in faults:
        if f.kind in ("noise", "dropout"):
            sel = (w.index >= f.start) & (w.index < f.end)
            cols = [(f.space, m) for m in an.METRICS] if f.metric == "*" else [(f.space, f.metric)]
            pd.testing.assert_frame_equal(t4.loc[sel, cols], w.loc[sel, cols])
