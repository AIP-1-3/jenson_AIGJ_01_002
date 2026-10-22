import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from tsinfer.models import forecast as fc  # noqa: E402
from tsinfer.models import forecast_ens as fe  # noqa: E402
from tests.test_forecast import CFG, _wide  # noqa: E402


def test_fit_cell_recovers_alpha_and_weight():
    rng = np.random.default_rng(0)
    cur = rng.normal(600, 50, 500)
    true_delta = rng.normal(0, 40, 500)
    truth = cur + true_delta
    dg = true_delta * 2.0                       # GRU 변화량이 두 배로 과장 → α 0.5
    assert fe.fit_cell(dg, None, cur, truth, use_w=False) == (0.0, 0.5)
    dc = true_delta + rng.normal(0, 1, 500)     # Chronos 가 거의 정답 → w 1, α 1
    w, a = fe.fit_cell(rng.normal(0, 40, 500), dc, cur, truth, use_w=True)
    assert w == 1.0 and a == 1.0


def test_fit_cell_all_nan_falls_back_to_model():
    nan = np.full(5, np.nan)
    assert fe.fit_cell(nan, None, nan, nan, use_w=False) == (0.0, 1.0)


@pytest.fixture(scope="module")
def members_and_wide():
    w = _wide(days=22)
    cfg = copy.deepcopy(CFG)
    cfg["model"]["epochs"] = 2
    return fe.train_members(cfg, w, None, lookbacks=(30, 60), seeds=(0, 1), log=lambda s: None), w, cfg


def test_members_gather_and_variants(members_and_wide):
    members, w, cfg = members_and_wide
    assert [(m["lookback"], m["seed"]) for m in members] == [(30, 0), (30, 1), (60, 0), (60, 1)]
    step = cfg["forecast"]["step_minutes"]
    g = fe.gather(members, "test", w, step)
    # 짧은 창 구성원이 만든 열쇠가 합집합에 다 들어 있고, 긴 창 구성원은 그 일부에만 값이 있다
    assert len(g["keys"]) == len(members[0]["test"]["co2"])
    assert g["stack"]["co2"].shape == (4, len(g["keys"]), 12)
    # 현재값·정답이 원래 표와 같다
    t, sp = g["keys"][100]
    assert g["cur"]["co2"][100] == pytest.approx(w.loc[t, (sp, "co2")])
    assert g["truth"]["co2"][100, 11] == pytest.approx(w.loc[t + pd.Timedelta(minutes=60), (sp, "co2")])

    g_va = fe.gather(members, "valid", w, step)
    chron = {m: g_va["truth"][m] - g_va["cur"][m][:, None] for m in fc.TARGETS}   # 정답을 아는 가짜 Chronos
    params = fe.fit_params(g_va, chron, "ens_chronos", 0)
    assert params[("lab", "co2")][-1][0] == 1.0          # 정답을 섞는 비율이 최대
    pred = fe.apply_params(g_va, chron, "ens_chronos", 0, params)
    rows = fe.mae_rows(g_va, pred, "ens_chronos", [60], step)
    assert (rows["ens_chronos"] < 1e-6).all()
    for v in ("single", "ens", "ens_space"):
        s = fe.crossfit_score(g_va, None, v, 0, "Asia/Seoul", step)
        assert np.isfinite(s)
    # 공통 α 방식은 공간마다 같은 계수
    shared = fe.fit_params(g_va, None, "ens", 0)
    assert shared[("lab", "co2")] == shared[("meet", "co2")]


def test_chronos_deltas_uses_same_keys(members_and_wide):
    members, w, cfg = members_and_wide
    g = fe.gather(members, "test", w, cfg["forecast"]["step_minutes"])
    calls = []

    def fake(ctx, H):
        calls.append(ctx.shape)
        last = ctx[:, -1]
        return np.repeat(np.stack([last - 1, last, last + 1], axis=-1)[:, None, :], H, axis=1)

    d = fe.chronos_deltas(g, w, fake, ctx_len=24, H=12)
    assert d["co2"].shape == (len(g["keys"]), 12)
    assert np.allclose(d["co2"], 0, atol=1e-3)            # 마지막 값 = 현재값 → 변화량 0
    assert all(c[1] == 24 for c in calls)
