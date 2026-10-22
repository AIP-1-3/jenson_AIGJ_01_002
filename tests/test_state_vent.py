import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from test_state import CFG, SCHED, _wide  # noqa: E402  (같은 합성 데이터)
from test_state_deep import _tiny_gru  # noqa: E402

from tsinfer.models import state as st  # noqa: E402
from tsinfer.models import state_vent as sv  # noqa: E402


def test_seasonal_features_look_only_at_the_past():
    w = _wide()
    sp = w.columns.get_level_values("space")[0]
    f = sv.seasonal_features(w, sp, "Asia/Seoul", 30, 1000.0)
    s = w[(sp, "co2")]
    t = w.index[400]
    day = pd.Timedelta(days=1)
    assert f.loc[t, "lag1d_now"] == pytest.approx(s.loc[t - day])
    want = s.loc[t - day + pd.Timedelta(minutes=5): t - day + pd.Timedelta(minutes=30)].max()
    assert f.loc[t, "lag1d_max30"] == pytest.approx(want)
    assert np.isnan(f["lag7d_now"].iloc[0])                         # 7일 전이 없으면 비어 있음
    # 미래 값을 바꿔도 지금 특성은 그대로 (누설 없음)
    w2 = w.copy()
    w2.loc[w.index > t, (sp, "co2")] += 500
    f2 = sv.seasonal_features(w2, sp, "Asia/Seoul", 30, 1000.0)
    pd.testing.assert_series_equal(f.loc[:t].iloc[-1], f2.loc[:t].iloc[-1])


def test_add_features_keeps_rows_and_runs_alarm(tmp_path):
    w = _wide()
    df = st.dataset(w, CFG, SCHED)
    aug = sv.add_features(df, w, CFG)
    assert len(aug) == len(df) and (aug.index == df.index).all()
    assert set(sv.NEW_FEATURES) <= set(aug.columns)
    gru = _tiny_gru(tmp_path / "gru.pt", w)
    before = sv.sd.forecast_alarm(CFG, w, df, gru, folds=3, recall_target=0.87)
    after = sv.sd.forecast_alarm(CFG, w, aug, gru, folds=3, recall_target=0.87)
    assert after["table"]["n"].iloc[0] == before["table"]["n"].iloc[0]     # 같은 시험 시점
    text = "\n".join(sv.build_report(CFG, before, after, 0.87))
    assert "## 1. 기준 확인" in text and "## 해석 (직접 작성)" in text
