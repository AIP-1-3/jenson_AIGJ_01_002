import numpy as np
import pandas as pd
import pytest

from tsinfer.models import anomaly as an
from tsinfer.models import anomaly_ctx as ac
from tsinfer.models import anomaly_deep as ad
from tests.test_anomaly_deep import CFG, SPACES, _persistence_gru, _wide


def test_stuck_ctx_uses_shorter_limit_in_use_hours():
    w = _wide(days=10)
    w[("lab", "temperature")] = w[("lab", "temperature")].round(0)    # 분해능이 거칠어 빈 시간엔 오래 그대로
    table = ac.fit_flat_ctx(w, SPACES, "Asia/Seoul", [])
    assert table[("lab", "temperature", 1)] <= table[("lab", "temperature", 0)]
    # 사용 시간에 3시간 값 고정 → 점수가 1 을 크게 넘는다
    t0 = pd.Timestamp("2026-09-08 01:00", tz="UTC")     # 한국 10시 (화요일)
    idx = w.index[(w.index >= t0) & (w.index < t0 + pd.Timedelta(hours=3))]
    w2 = w.copy()
    w2.loc[idx, ("lab", "co2")] = w2.loc[t0, ("lab", "co2")]
    sc = ac.stuck_ctx_scores(w2, SPACES, table, "Asia/Seoul", [])["lab"]["co2"]
    assert sc.loc[idx[-1]] > 3 and sc.loc[idx[-1]] > sc.drop(idx).max()


def test_stuck2_use_is_blank_outside_use_hours():
    w = _wide(days=6)
    move = ad.fit_move_scale(w, SPACES)
    sc = ac.stuck2_use_scores(w, SPACES, move, "Asia/Seoul", [])["lab"]
    use = ad.usage_mask(w.index, "Asia/Seoul", [])
    assert sc[~use].isna().all().all()


def test_scorer_matches_reference_scoring():
    w = _wide(days=8)
    inj, faults = an.inject(w, SPACES, per_kind=2, seed=0)
    flags = an.method_flags("rules", inj, SPACES, None)
    anyf = ad.any_flags(flags)
    sc = ac.Scorer(w.index, faults, 8.0)
    assert np.array_equal(sc.hits(anyf), ad.fault_hits(anyf, w.index, faults))
    assert sc.fa(anyf) == pytest.approx(ad.false_alarms(anyf, w.index, faults, 8.0))
    assert ac.Scorer(w.index, [], 8.0).fa(anyf) == pytest.approx(ad.false_alarms(anyf, w.index, [], 8.0))


def test_full_run_picks_on_validation_5_kinds():
    pytest.importorskip("torch")
    w = _wide()
    gru = _persistence_gru(w)
    from tsinfer.data.split import split_by_time

    train = split_by_time(w, CFG)[0]
    ae = an.fit_ae(train, SPACES, epochs=2, threads=2)
    ae.threshold = an.pick_threshold(ae, split_by_time(w, CFG)[1], SPACES)
    out = ac.run(CFG, w, ae, gru=gru, per_kind=3, log=lambda *a: None)
    assert set(out["names"]) == {"pe", "cusum", "floor", "cross", "stuck2", "stuck_ctx", "stuck2_use", "pe_long", "high"}
    assert len(out["combos"]) == 3 ** len(out["names"])
    best = out["combos"][out["combos"]["combo"] == out["best"]].iloc[0]
    assert best["fa"] <= ac.FA_LIMIT
    assert {f.kind for f in out["v5_faults"]} <= set(ad.T4_KINDS)
    text = "\n".join(ac.build_report(out, 3))
    for h in ("## 1. 요약", "## 2. 과제 안내 5종", "## 5. 조합 선택", "## 해석 (직접 작성)"):
        assert h in text
