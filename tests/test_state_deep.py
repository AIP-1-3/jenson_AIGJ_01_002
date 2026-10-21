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
