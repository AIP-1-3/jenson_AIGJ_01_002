import copy
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tsinfer.models import forecast as fc  # noqa: E402
from tsinfer.models import forecast_deep as fd  # noqa: E402
from tests.test_forecast import CFG, _wide  # noqa: E402


@pytest.mark.parametrize("arch", fc.ARCHS)
def test_arch_forward_and_save_load(arch, tmp_path):
    w = _wide(days=6, start="2026-09-12")
    spec = fc.fit_spec(w, w.index[:1000], CFG)
    mc = {**CFG["model"], "arch": arch, "tcn_channels": 16}
    model = fc.build_model(spec, mc)
    x, _, _ = fc.make_windows(w, spec)
    out = model(torch.from_numpy(x[:5]))
    assert out.shape == (5, 2 * len(spec.horizons))
    if arch == "tcn":   # 인과성: 마지막 칸만 바꾸면 출력이 변하고, 그 앞 칸 출력은 미래 칸 영향이 없어야
        model.eval()
        xx = torch.from_numpy(x[:2]).clone()
        y0 = model.blocks(xx.transpose(1, 2))
        xx[:, -1] += 1.0
        y1 = model.blocks(xx.transpose(1, 2))
        assert torch.allclose(y0[:, :, :-1], y1[:, :, :-1])
    path = tmp_path / f"{arch}.pt"
    fc.save(model, spec, mc, path)
    m2, _ = fc.load(path)
    assert type(m2).__name__ == type(model).__name__
    m2.eval(), model.eval()
    assert torch.allclose(m2(torch.from_numpy(x[:3])), model(torch.from_numpy(x[:3])))


def test_old_checkpoint_without_arch_loads_as_gru(tmp_path):
    w = _wide(days=6, start="2026-09-12")
    spec = fc.fit_spec(w, w.index[:1000], CFG)
    model = fc.build_model(spec, CFG["model"])
    path = tmp_path / "old.pt"
    torch.save({"state_dict": model.state_dict(), "spec": spec.to_dict(), "model_cfg": CFG["model"]}, path)
    m2, _ = fc.load(path)
    assert type(m2).__name__ == "GRUForecast"


@pytest.fixture(scope="module")
def arch_run():
    return fd.run_arch(copy.deepcopy(CFG), _wide(), epochs=2, patience=2, log=lambda *_: None)


@pytest.fixture(scope="module")
def loso_run():
    return fd.run_loso(copy.deepcopy(CFG), _wide(), epochs=2, patience=2, log=lambda *_: None)


def test_arch_stage_short_training(arch_run):
    res, stage2, skipped = arch_run
    assert [r["arch"] for r in res[:3]] == list(fc.ARCHS) and stage2 in fc.ARCHS and not skipped
    assert {r["lookback_min"] for r in res if r["arch"] == stage2} == {60, 120, 240}
    assert len(res) == 5 and all(np.isfinite(r["score"]) and r["params"] > 0 for r in res)
    summ, tab = fd.arch_tables(res)
    assert len(summ) == 5 and {"CO₂ 60분", "온도 60분", "60분 개선 %"} <= set(tab.columns)


def test_loso_holdout_never_in_training(loso_run):
    cfg = copy.deepcopy(CFG)
    cfg["model"]["epochs"] = 2
    w = _wide()
    r = fd.fit_eval(cfg, w, holdout="meet", log=lambda *_: None)
    spec = r["spec"]
    assert r["train_spaces"] == ["lab"] and spec.spaces == ["lab"]
    # 정규화 통계는 학습 공간 평균으로 대신 (빼는 공간 값으로 계산하지 않음)
    assert spec.mean["meet|co2"] == spec.mean["lab|co2"]
    assert spec.target_std["meet|co2"] == spec.target_std["lab|co2"]
    assert set(r["val"]["space"]) == {"lab"} and set(r["test"]["space"]) == {"meet"}
    assert (r["test"]["model"] > 0).all()

    t, ref, info, skipped = loso_run
    assert set(t["빼고 시험한 공간"]) == {"lab", "meet"} and not skipped
    assert t["포함 학습"].notna().all() and len(t) == 2 * 2 * 4


def test_report_renders(arch_run, loso_run):
    res, stage2, _ = arch_run
    args = SimpleNamespace(full_grid=False, epochs=1, patience=1, loso_epochs=1, loso_patience=1,
                           loso_arch="gru", loso_lookback=120)
    text = fd.write_report(CFG, _wide(), None, args, res, stage2, [], loso_run, 60.0)
    assert "## 측정 조건" in text and text.rstrip().splitlines()[0].startswith("# 과제 3 심화")
    assert "## 해석 (직접 작성)" in text and "(B) 한 공간 빼기" in text
