import numpy as np
import pandas as pd

from tsinfer.service import control_sim as cs

CFG = {
    "spaces": ["lab", "outside"],
    "metrics": ["co2", "temperature", "humidity"],
    "split": {"train_end": "2026-09-10", "valid_end": "2026-09-20"},
    "forecast": {"horizon_minutes": 60, "step_minutes": 5, "report_horizons": [5, 15, 30, 60]},
    "thresholds": {"co2_ventilate_ppm": 1000},
}


def _room(days=30, tau=90.0, gen=50.0, seed=0):
    """9~12시에 사람이 있어 5분마다 gen ppm 발생, 나머지는 τ 로 자연 감쇠."""
    idx = pd.date_range("2026-09-01", periods=days * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(seed)
    a = np.exp(-5 / tau)
    c = np.empty(len(idx))
    c[0] = 450
    occ = (idx.hour >= 9) & (idx.hour < 12)
    for t in range(1, len(idx)):
        c[t] = 420 + (c[t - 1] - 420) * a + (gen if occ[t - 1] else 0) + rng.normal(0, 1)
    cols = pd.MultiIndex.from_product([["lab", "outside"], ["co2", "temperature", "humidity"]])
    wide = pd.DataFrame(np.nan, index=idx, columns=cols)
    wide[("lab", "co2")] = c
    wide[("lab", "temperature")] = 23.0
    return wide


def test_estimate_tau_recovers_decay():
    wide = _room(days=10, tau=90.0)
    taus = cs.estimate_tau(wide[("lab", "co2")], 5)
    assert len(taus) >= 5 and 70 < np.median(taus) < 110


def test_no_control_replays_observed_and_reactive_reduces():
    obs = _room(days=3)[("lab", "co2")].to_numpy().copy()
    obs[100:103] = np.nan                                     # 결측 구간은 실측으로 다시 시작
    a_n, a_v = np.exp(-5 / 90), np.exp(-5 / 20)
    c, vent = cs.simulate(obs, a_n, a_v, "none")
    ok = np.isfinite(obs)
    assert np.allclose(c[ok], obs[ok]) and not vent.any()
    c_r, v_r = cs.simulate(obs, a_n, a_v, "reactive")
    assert (c_r[ok] > 1000).sum() < (obs[ok] > 1000).sum() and v_r.any()
    # 완벽 예측을 쓴 예측형은 반응형보다 초과가 적거나 같다
    oracle = cs.oracle_predictions(pd.Series(obs), 12)
    c_p, _ = cs.simulate(obs, a_n, a_v, "predictive", oracle, lead_steps=6)
    assert (c_p[ok] > 1000).sum() <= (c_r[ok] > 1000).sum()


def test_run_falls_back_to_trend_without_model(tmp_path):
    out = cs.run(CFG, _room(days=30), days=7, model_path=tmp_path / "none.pt", log=lambda *_: None)
    res = out["results"]
    assert out["spaces"] == ["lab"] and "추세" in out["model"]
    assert set(res["tau_vent_min"]) == set(cs.TAU_VENT)
    main = res[res["tau_vent_min"] == cs.TAU_VENT_MAIN].set_index("policy")["exceed_min"]
    assert main["제어 없음(실측)"] > main["반응형"] >= main["예측형: 완벽 예측(상한)"]
    assert res["replay_err_ppm"].dropna().max() < 1e-6
    text = cs.report(out, 7, 1.0)
    assert "## 해석 (직접 작성)" in text and "추세 기준선(대체)" in text
