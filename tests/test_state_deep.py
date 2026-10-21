import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from test_state import CFG, SCHED, _wide  # noqa: E402  (같은 합성 데이터)

from tsinfer.models import state_deep as sd  # noqa: E402


def test_runs():
    idx = pd.date_range("2026-10-06", periods=8, freq="5min", tz="UTC")
    r = sd.runs(pd.Series([0, 1, 1, 0, 1, 1, 1, 0], index=idx).astype(bool))
    assert [n for _, _, n in r] == [2, 3] and r[1][0] == idx[4]
    gap = idx.delete(2)                                       # 시점이 비면 다른 구간
    assert len(sd.runs(pd.Series(True, index=gap))) == 2


def test_schedule_mismatch_rules():
    idx = pd.date_range("2026-10-06 00:00", "2026-10-06 03:55", freq="5min", tz="UTC")   # KST 09:00~12:55
    sched = pd.Series(((idx.hour >= 0) & (idx.hour < 2)).astype(float), index=idx)       # 수업 KST 9~11시
    est = pd.Series(1.0, index=idx)
    est[(idx >= "2026-10-06 01:00") & (idx < "2026-10-06 01:20")] = 0     # 10시 칸 시작 20분 비어 있음 → 1건
    est[(idx >= "2026-10-06 02:00") & (idx < "2026-10-06 03:00")] = 0     # 수업 끝 → 비어 있음
    est[idx >= "2026-10-06 03:40"] = 1                                       # 12:40~ 사용 (20분) → 1건
    est[(idx >= "2026-10-06 03:00") & (idx < "2026-10-06 03:40")] = 0
    ev = sd.schedule_mismatch(est, sched, "Asia/Seoul")
    ns = ev[ev["kind"] == "수업인데 비어 있음"]
    un = ev[ev["kind"] == "수업 아닌데 사용"]
    assert len(ns) == 1 and ns.iloc[0]["start"].hour == 10
    assert len(un) == 1 and un.iloc[0]["minutes"] == 20
    c = sd.mismatch_counts({"rule": ev}, days=1.0)
    assert c["events"].tolist() == [1, 1]


def test_loso_and_mismatch_on_synthetic():
    tables, events, df = sd.run(CFG, _wide(), SCHED, space="lab", models=("hgb",), folds=3)   # 시간 절약
    occ = tables["occupied"]
    assert set(occ["space"]) == {"lab", "meet"}
    assert {"규칙", "hgb (이 공간 빼고)", "hgb (같은 공간 포함)"} <= set(occ["method"])
    held = occ[(occ["space"] == "lab") & (occ["method"] == "hgb (이 공간 빼고)")].iloc[0]
    assert held["n"] > 0 and 0 <= held["macro_f1"] <= 1
    summ = sd.loso_summary(occ)
    assert "macro_f1_drop" in summ.columns
    assert "ventilation" in tables
    assert set(events) == {"규칙", "hgb (lab 빼고 학습)"}
    assert set(sd.mismatch_counts(events, 5)["kind"]) == {"수업인데 비어 있음", "수업 아닌데 사용"}


# ---------------------------------------------------------------- 4절 환기 경보 예측 기반 · 5절 PIR

def _tiny_gru(path, wide):
    """작은 무작위 GRU 를 forecast.save 형식으로 저장 (성능이 아니라 경로 확인용)."""
    pytest.importorskip("torch")
    from tsinfer.models import forecast as fc

    cfg = {**CFG, "model": {**CFG["model"], "lookback_minutes": 60, "hidden": 8, "layers": 1, "dropout": 0.0}}
    spec = fc.fit_spec(wide, wide.index[wide.index < CFG["split"]["train_end"]], cfg)
    fc.save(fc.build_model(spec, cfg["model"]), spec, cfg["model"], path=path)
    return path


def test_sweep_and_pick_two():
    y = np.array([0, 0, 0, 0, 1, 1, 0, 1])
    s = np.array([1, 2, 3, 4, 8, 9, 5, 7.0])
    t_f1, t_rec = sd.pick_two(y, s)
    assert ((s >= t_f1) == y.astype(bool)).all() and t_rec <= t_f1
    sw = sd.sweep(y, s, np.array([7.0]))
    assert sw.loc[0, "recall"] == 1 and sw.loc[0, "macro_f1"] == 1


def test_forecast_alarm_with_gru(tmp_path):
    from tsinfer.models import state as st

    w = _wide()
    df = st.dataset(w, CFG, SCHED)
    res = sd.forecast_alarm(CFG, w, df, _tiny_gru(tmp_path / "gru.pt", w), folds=3)
    t = res["table"]
    for m in ("규칙 (state.py: 30분 기울기 연장)", "기존 hgb (F1 최대 운영점)", "예측 기반 (F1 최대 운영점)",
              "예측 기반 (재현율 운영점)", "결합 hgb (F1 최대 운영점)", "결합 hgb (재현율 운영점)"):
        assert m in t.index and 0 <= t.loc[m, "macro_f1"] <= 1
    assert t["n"].nunique() == 1                      # 모든 행이 같은 시험 시점
    assert res["info"]["test_rows"] <= res["info"]["test_rows_state"]
    lines = sd.forecast_alarm_lines(res, CFG)
    assert any(x.startswith("- 확인 기준") for x in lines) and "### 해석 (직접 작성)" in lines


def test_forecast_alarm_without_gru_and_sections(tmp_path):
    from tsinfer.models import state as st

    w = _wide()
    df = st.dataset(w, CFG, SCHED)
    res = sd.forecast_alarm(CFG, w, df, tmp_path / "없는.pt", folds=3)
    t = res["table"]
    assert t.loc["예측 기반 (F1 최대 운영점)", "note"].startswith("없음")
    assert t.loc["결합 hgb (재현율 운영점)", "note"].startswith("없음")
    assert pd.notna(t.loc["기존 hgb (F1 최대 운영점)", "macro_f1"])
    assert t.loc["규칙 (state.py: 30분 기울기 연장)", "n"] == res["info"]["test_rows_state"]   # GRU 없으면 state.py 와 같은 시점
    lines = sd.extra_sections(CFG, w, SCHED, df, gru_path=tmp_path / "없는.pt", db=tmp_path / "없는.db",
                              state_model_path=tmp_path / "없는.joblib", folds=3)
    text = "\n".join(lines)
    assert "## 4. 환기 경보" in text and "## 5. 직접 관찰 대용" in text
    assert "데이터 없음" in text and "실행 오류" not in text


def _pir_db(path, wide, days=2, pir=True):
    from tsinfer.collector.store import SCHEMA

    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    part = wide[wide.index >= wide.index.max() - pd.Timedelta(days=days)]
    rows = []
    for (sp, m), s in part.items():
        for t, v in s.dropna().items():
            rows.append((sp, "d1", m, t.isoformat(), float(v), t.isoformat(), 1, "test"))
    if pir:
        local = part.index.tz_convert("Asia/Seoul")
        occ = (local.dayofweek < 5) & (local.hour >= 9) & (local.hour < 12)
        for t, o in zip(part.index[::2], occ[::2]):                  # 10분 간격 보고
            rows.append(("lab", "am107", "pir_activity", t.isoformat(), float(3 if o else 0), t.isoformat(), 1, "test"))
    conn.executemany("INSERT OR IGNORE INTO readings VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return path


def test_pir_compare_missing(tmp_path):
    from tsinfer.models import state as st

    w = _wide(days=6)
    df = st.dataset(w, CFG, SCHED)
    r = sd.pir_compare(CFG, SCHED, w, df, db=tmp_path / "없는.db", space="lab")
    assert "missing" in r and sd.pir_lines(r)[2].startswith("데이터 없음")
    r = sd.pir_compare(CFG, SCHED, w, df, db=_pir_db(tmp_path / "nopir.db", w, pir=False), space="lab", since=None)
    assert "missing" in r


def test_pir_compare_synthetic(tmp_path):
    from tsinfer.models import state as st

    w = _wide()
    df = st.dataset(w, CFG, SCHED)
    db = _pir_db(tmp_path / "t.db", w, days=3)
    cfg = {**CFG, "preprocess": {"resample": "5min", "short_gap_max_minutes": 10,
                                 "valid_range": {"co2": [350, 5000], "temperature": [-10, 50], "humidity": [0, 100]}}}
    r = sd.pir_compare(cfg, SCHED, w, df, db=db, space="lab", since=None,
                       state_model_path=tmp_path / "없는.joblib")
    t = r["table"]
    assert "시간표" in t.index and "규칙" in t.index and any(i.startswith("hgb") for i in t.index)
    assert t.loc["시간표", "agree_%"] > 90                    # 합성 PIR 은 시간표대로 움직임
    assert set(r["counts"]["kind"]) == {"시간표=사용인데 PIR 없음", "시간표=빔인데 PIR 있음"}
    assert r["info"]["sensor_src"] == "수집기 DB" and r["info"]["devices"] == ["am107"]
    text = "\n".join(sd.pir_lines(r))
    assert "한계" in text and "데이터 없음" not in text


def test_pir_bins_spreads_interval():
    idx = pd.to_datetime(["2026-10-20 00:10", "2026-10-20 00:20", "2026-10-20 02:00"], utc=True)
    b = sd.pir_bins(pd.DataFrame({"ts": [t.isoformat() for t in idx], "value": [0, 2, 1]}))
    assert b[pd.Timestamp("2026-10-20 00:10", tz="UTC")] == 1 and b[pd.Timestamp("2026-10-20 00:15", tz="UTC")] == 1
    assert b[pd.Timestamp("2026-10-20 00:05", tz="UTC")] == 0
    assert pd.Timestamp("2026-10-20 01:00", tz="UTC") not in b.index   # 30분 넘게 빈 보고 사이는 '모름'
