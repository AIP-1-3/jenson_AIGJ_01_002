from pathlib import Path

from tsinfer.viz.report_tables import read_tables
from tsinfer.viz.results import build, captions, render

ROOT = Path(__file__).resolve().parents[1]


def test_tuple_headers_parse(tmp_path):
    md = tmp_path / "x.md"
    md.write_text("# 결과\n\n|  | ('gru', 5) | ('gru', 60) |\n|:--|--:|--:|\n| ('실습실', 'co2') | 12.0 | 31.82 |\n",
                  encoding="utf-8")
    t = read_tables(md)["결과"][0]
    assert t.columns.tolist() == [("gru", 5), ("gru", 60)]
    assert t.index.tolist() == [("실습실", "co2")]
    assert t.iat[0, 1] == 31.82


def test_summary_numbers_come_from_reports():
    """요약 페이지 숫자가 보고서 표와 같다 (문서끼리 숫자 일치)."""
    d = build(ROOT)
    v = {x["task"]: x for x in d["verdicts"]}
    goal = {r["space"]: r for r in d["t3"]["goal"]}
    text = (ROOT / "reports" / "03_forecast.md").read_text(encoding="utf-8")
    for space, r in goal.items():
        assert f"{r['improve']}" in text and f"−{r['improve']}%" in v["3"]["result"]
    assert d["t3"]["curve"]["epoch"][0] == 1
    assert v["7"]["status"] in ("pass", "fail")


def test_render_without_reports(tmp_path):
    (tmp_path / "reports").mkdir()
    d = build(tmp_path)
    page = render(d)
    assert "{{" not in page and page.startswith("<!doctype html>")
    assert captions(d) == {}
    assert not render(d, fragment=True).startswith("<!doctype")
