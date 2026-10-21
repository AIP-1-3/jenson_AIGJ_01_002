"""과제 2~7 결과 한 페이지 요약: python -m tsinfer.viz.results [--out reports/summary.html]

reports/*.md 의 표를 읽어 과제별 판정, 학습 곡선, 기준선 대비 오차, 이상 탐지·상태·엣지 결과를
그래프 한 페이지(HTML)로 만든다. 숫자는 보고서 표에서만 가져오므로, run_all.sh 로 보고서를 다시 만든 뒤
이 명령을 다시 돌리면 요약도 같은 숫자로 바뀐다.
logs/service_cycles.jsonl 이 있으면 과제 8(12시간 가동) 요약도 함께 넣는다.

그림 속 숫자는 그림마다 붙는 설명 글에도 적는다 (그림을 못 보는 채점·검색에서도 읽히게).
"""
from __future__ import annotations

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from tsinfer.config import ROOT
from tsinfer.viz.report_tables import bullets, find, read_tables

SPACES = ["실습실", "사무실", "회의실"]
HORIZONS = [5, 15, 30, 60]
TEMPLATE = Path(__file__).with_name("results_template.html")


def _r(x: Any, nd: int = 1) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


def cell(df: pd.DataFrame, row: Any, col: Any) -> Any:
    """튜플 머리글((공간, 항목) 등)도 그대로 찾는다. .loc 은 튜플을 다중 인덱스로 해석해 버린다."""
    rows, cols = list(df.index), list(df.columns)
    if row not in rows or col not in cols:
        return None
    return df.iat[rows.index(row), cols.index(col)]


def _verdict(task: str, title: str, target: str, result: str, status: str, link: str) -> dict:
    # status: pass | fail | partial | info
    return {"task": task, "title": title, "target": target, "result": result, "status": status, "link": link}


def task2(rep: Path) -> tuple[dict | None, dict | None]:
    t = find(read_tables(rep / "02_baselines.md"), "결과")
    if t is None:
        return None, None
    out = {}
    for space in SPACES:
        out[space] = {m: _r(cell(t, (space, "co2", m), 60.0)) for m in ("persistence", "mean_1h", "yesterday")
                      if (space, "co2", m) in t.index}
    best_all = all(min(v, key=lambda k: v[k] if v[k] is not None else 1e9) == "persistence" for v in out.values() if v)
    v = _verdict("2", "기준선", "기준선 3종 × 5·15·30·60분 MAE",
                 "현재값 유지가 세 공간 모두 최선" if best_all else "공간마다 최선 기준선이 다름",
                 "info", "reports/02_baselines.md")
    return v, {"mae60": out}


def task3(rep: Path) -> tuple[dict | None, dict | None]:
    tables = read_tables(rep / "03_forecast.md")
    goal, mae, curve = find(tables, "목표 확인"), find(tables, "결과 MAE"), find(tables, "학습 곡선")
    if goal is None or mae is None:
        return None, None
    rows = goal.to_dict("records")
    passed = sum(bool(r.get("pass(≥30%)")) for r in rows)
    result = ", ".join(f"{r['space']} −{_r(r['improve_%'])}%" for r in rows)
    v = _verdict("3", "60분 예측 (GRU)", "60분 CO₂ 현재값 유지 대비 −30%", result,
                 "pass" if passed == len(rows) else ("partial" if passed else "fail"), "reports/03_forecast.md")
    by_h = {}
    for space in SPACES:
        key = (space, "co2")
        if key not in mae.index:
            continue
        by_h[space] = {m: [_r(cell(mae, key, (m, h))) for h in HORIZONS]
                       for m in ("persistence", "gru", "gru_no_time", "gru_no_schedule")}
    data = {"goal": [{"space": r["space"], "persistence": _r(r["persistence"]), "gru": _r(r["gru"]),
                      "improve": _r(r["improve_%"])} for r in rows],
            "by_horizon": by_h,
            "curve": None if curve is None else {
                "epoch": [int(e) for e in curve["epoch"]],
                "train": [_r(x, 4) for x in curve["train"]],
                "valid": [_r(x, 4) for x in curve["valid"]],
                "sec": [_r(x) for x in curve["sec"]]},
            "conditions": bullets(rep / "03_forecast.md", "측정 조건")}
    return v, data


def task4(rep: Path) -> tuple[dict | None, dict | None]:
    tables = read_tables(rep / "04_anomaly.md")
    parts = {"규칙": find(tables, "규칙"), "오토인코더": find(tables, "오토인코더"), "결합": find(tables, "결합")}
    if any(p is None for p in parts.values()):
        return None, None
    comb = parts["결합"].set_index("kind")
    det, fa = _r(comb.loc["전체", "detection_rate"]), _r(comb.loc["전체", "false_alarms_per_day"])
    ok_det, ok_fa = det is not None and det >= 80, fa is not None and fa <= 1
    v = _verdict("4", "이상 탐지", "탐지율 ≥80%, 헛경보 ≤1회/일", f"결합 탐지율 {det}%, 헛경보 {fa}회/일",
                 "pass" if ok_det and ok_fa else ("partial" if ok_det or ok_fa else "fail"), "reports/04_anomaly.md")
    kinds = [k for k in comb.index if k != "전체"]
    data = {"kinds": kinds,
            "detection": {name: [_r(df.set_index("kind").loc[k, "detection_rate"]) for k in kinds]
                          for name, df in parts.items()},
            "total": {name: {"detection": _r(df.set_index("kind").loc["전체", "detection_rate"]),
                             "false_alarms": _r(df.set_index("kind").loc["전체", "false_alarms_per_day"])}
                      for name, df in parts.items()}}
    return v, data


def task5(rep: Path) -> tuple[dict | None, dict | None]:
    tables = read_tables(rep / "05_state.md")
    occ, vent = find(tables, "사용 여부"), find(tables, "환기 경보")
    if occ is None or vent is None:
        return None, None
    models = [m for m in vent.index if m != "rule"]
    best = max(models, key=lambda m: vent.loc[m, "recall"])
    rec, f1, f1_rule = _r(vent.loc[best, "recall"], 3), _r(vent.loc[best, "macro_f1"], 3), _r(vent.loc["rule", "macro_f1"], 3)
    ok_rec, ok_f1 = rec >= 0.85, f1 > f1_rule
    v = _verdict("5", "상태 추정", "환기 경보 재현율 ≥0.85, 규칙보다 macro-F1↑",
                 f"{best} 재현율 {rec}, macro-F1 {f1} (규칙 {f1_rule})",
                 "pass" if ok_rec and ok_f1 else ("partial" if ok_rec or ok_f1 else "fail"), "reports/05_state.md")
    pick = ["recall", "precision", "macro_f1"]
    data = {"vent": {m: [_r(vent.loc[m, c], 3) for c in pick] for m in vent.index},
            "occ": {m: [_r(occ.loc[m, c], 3) for c in pick] for m in occ.index},
            "metrics": pick}
    return v, data


def task6(rep: Path) -> tuple[dict | None, dict | None]:
    tables = read_tables(rep / "06_pretrained.md")
    mae, lat, mem, cov = (find(tables, "1. 정확도"), find(tables, "3. 추론 시간"), find(tables, "4. 메모리"),
                          find(tables, "2. 10~90%"))
    if mae is None:
        return None, None
    m60 = {s: {m: _r(cell(mae, (s, "co2"), (m, 60))) for m in ("persistence", "gru", "chronos")}
           for s in SPACES if (s, "co2") in mae.index}
    lat1 = _r(lat.set_index("batch").loc[1.0, "mean_ms"]) if lat is not None else None
    mem_mb = int(mem.loc["model_increase_mb", "MB"]) if mem is not None else None
    cov60 = [_r(cell(cov, (s, "co2"), 60.0)) for s in SPACES if cov is not None and (s, "co2") in cov.index]
    lab = m60.get("실습실", {})
    v = _verdict("6", "사전학습 (Chronos-Bolt)", "정확도·구간 적중률·지연·메모리 비교",
                 f"실습실 60분 {lab.get('chronos')} vs GRU {lab.get('gru')} ppm, 지연 {lat1}ms, 메모리 +{mem_mb}MB",
                 "info", "reports/06_pretrained.md")
    return v, {"mae60": m60, "latency_ms": lat1, "memory_mb": mem_mb,
               "coverage60": [min(cov60), max(cov60)] if cov60 else None}


def task7(rep: Path) -> tuple[dict | None, dict | None]:
    tables = read_tables(rep / "07_edge.md")
    lat, gru, ae, teg = (find(tables, "지연"), find(tables, "예측 GRU"), find(tables, "오토인코더"),
                         find(tables, "측정 중 장비"))
    if lat is None or gru is None or ae is None:
        return None, None
    diff = _r(gru.loc["tensorrt_fp16", "max_abs_diff_ppm"], 2)
    flag = _r(ae.loc["tensorrt_fp16", "flag_change_%"], 2)
    ok = diff <= 1 and flag <= 0.1
    latency: dict[str, dict[str, float | None]] = {}
    for (model, backend) in lat.index:
        x = _r(cell(lat, (model, backend), ("mean_ms", 1)), 3)
        if x is not None:
            latency.setdefault(model, {})[backend] = x
    v = _verdict("7", "엣지 최적화", "변환 후 CO₂ 차이 ≤1ppm, 판정 변화 ≤0.1%",
                 f"TensorRT FP16 {diff}ppm, 판정 변화 {flag}%", "pass" if ok else "fail", "reports/07_edge.md")
    power = None
    if teg is not None:
        t = teg["value"] if "value" in teg else teg.iloc[:, 0]
        power = {k: int(t.get(k)) for k in ("power_vdd_in_mean_mw", "power_vdd_in_max_mw", "ram_used_max_mb")}
    return v, {"latency": latency, "power": power}


def task8(root: Path, hours: float) -> dict | None:
    path = root / "logs" / "service_cycles.jsonl"
    if not path.exists():
        return _verdict("8", "상시 서비스", "12시간 무중단, 수집 10분 중단 시 끊김 경보",
                        "logs/service_cycles.jsonl 없음 (Jetson 에서 다시 만들면 채워진다)", "info", "docs/interface.md")
    from tsinfer.service.summary import summarize

    s = summarize(path, hours)
    if not s.get("cycles"):
        return None
    ok = s["failed_cycles"] == 0 and s["max_gap_min"] <= 2
    return _verdict("8", "상시 서비스", f"{hours:g}시간 무중단",
                    f"주기 {s['cycles']}/{s['expected_cycles']}, 실패 {s['failed_cycles']}, 최대 공백 {s['max_gap_min']}분, "
                    f"p95 {s['p95_duration_ms']}ms", "pass" if ok else "fail", "docs/interface.md")


def build(root: Path = ROOT, hours: float = 12) -> dict:
    rep = root / "reports"
    verdicts, data = [], {}
    for name, fn in (("t2", task2), ("t3", task3), ("t4", task4), ("t5", task5), ("t6", task6), ("t7", task7)):
        try:
            v, d = fn(rep)
        except FileNotFoundError:
            v, d = None, None
        if v:
            verdicts.append(v)
        data[name] = d
    if (v8 := task8(root, hours)):
        verdicts.append(v8)
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdicts": verdicts, **data}


# ── 그림 설명 글 (그림 속 숫자를 본문에 한 번 더) ─────────────────────────
def captions(d: dict) -> dict[str, str]:
    c: dict[str, str] = {}
    if d.get("t3") and d["t3"].get("curve"):
        cv = d["t3"]["curve"]
        best = min(range(len(cv["valid"])), key=lambda i: cv["valid"][i])
        c["curve"] = (f"에폭 {len(cv['epoch'])}회 학습. 학습 손실은 {cv['train'][0]} → {cv['train'][-1]}, "
                      f"검증 손실은 에폭 {cv['epoch'][best]} 에서 가장 낮은 {cv['valid'][best]} 이었다"
                      + (". 그 뒤로는 나아지지 않아 학습을 멈췄고 이 에폭의 가중치를 썼다. " if best < len(cv["valid"]) - 1 else ". ")
                      + f"평균 에폭 시간 {round(sum(cv['sec']) / len(cv['sec']), 1)}초 (Jetson CPU 3스레드).")
    if d.get("t3"):
        g = d["t3"]["goal"]
        c["goal"] = "60분 CO₂ MAE (같은 시험 시점): " + ", ".join(
            f"{r['space']} 현재값 유지 {r['persistence']} → GRU {r['gru']} ppm (−{r['improve']}%)" for r in g) + \
            f". 목표 −30% 를 넘은 공간 {sum(1 for r in g if (r['improve'] or 0) >= 30)}/{len(g)}곳."
        bh = d["t3"]["by_horizon"]
        c["horizon"] = "지평이 길수록 GRU 와 현재값 유지의 차이가 벌어진다. " + "; ".join(
            f"{s} 5분 {v['gru'][0]} vs {v['persistence'][0]}, 60분 {v['gru'][-1]} vs {v['persistence'][-1]} ppm"
            for s, v in bh.items()) + "."
    if d.get("t2"):
        c["baseline"] = "과제 2 기준선 60분 CO₂ MAE: " + "; ".join(
            f"{s} 현재값 유지 {v.get('persistence')}, 1시간 평균 {v.get('mean_1h')}, 어제 같은 시각 {v.get('yesterday')} ppm"
            for s, v in d["t2"]["mae60"].items()) + ". 시험 시점 전체 기준이라 과제 3 의 같은 시점 비교 값과 다를 수 있다."
    if d.get("t4"):
        t4 = d["t4"]
        c["anomaly"] = "고장 종류별 15건씩 넣었을 때 탐지율(%): " + "; ".join(
            f"{k} 규칙 {t4['detection']['규칙'][i]} · 오토인코더 {t4['detection']['오토인코더'][i]} · 결합 {t4['detection']['결합'][i]}"
            for i, k in enumerate(t4["kinds"])) + ". 헛경보(회/일): " + ", ".join(
            f"{n} {v['false_alarms']}" for n, v in t4["total"].items()) + ". 결합 탐지율이 80% 미만인 종류: " + (", ".join(
            k for i, k in enumerate(t4["kinds"]) if (t4["detection"]["결합"][i] or 0) < 80) or "없음") + "."
    if d.get("t5"):
        t5 = d["t5"]
        c["state"] = "환기 경보(30분 전) 재현율·정밀도·macro-F1: " + "; ".join(
            f"{m} {v[0]} / {v[1]} / {v[2]}" for m, v in t5["vent"].items()) + \
            ". 확인 기준은 재현율 0.85 이상이면서 macro-F1 이 규칙보다 높은 것."
    if d.get("t6"):
        t6 = d["t6"]
        c["chronos"] = "과제 6 평가 시점 300개의 60분 CO₂ MAE: " + "; ".join(
            f"{s} 현재값 유지 {v['persistence']}, GRU {v['gru']}, Chronos {v['chronos']} ppm" for s, v in t6["mae60"].items()) + \
            f". Chronos 지연 {t6['latency_ms']}ms(배치 1), 모델 메모리 +{t6['memory_mb']}MB" + \
            (f", 10~90% 구간 적중률 {t6['coverage60'][0]}~{t6['coverage60'][1]}%" if t6.get("coverage60") else "") + "."
    if d.get("t7"):
        t7 = d["t7"]
        c["edge"] = "배치 1 평균 지연(ms, 로그 눈금): " + "; ".join(
            f"{m} " + ", ".join(f"{b} {x}" for b, x in v.items()) for m, v in t7["latency"].items()) + "."
        if t7.get("power"):
            p = t7["power"]
            c["edge"] += f" 측정 중 전력 평균 {p['power_vdd_in_mean_mw']}mW, 최대 {p['power_vdd_in_max_mw']}mW, RAM 최대 {p['ram_used_max_mb']}MB (tegrastats)."
    return c


def render(d: dict, fragment: bool = False) -> str:
    page = TEMPLATE.read_text(encoding="utf-8")
    cap = captions(d)
    rows = []
    label = {"pass": "통과", "fail": "미달", "partial": "일부 통과", "info": "비교"}
    for v in d["verdicts"]:
        rows.append(
            f'<tr><td class="num">{html.escape(v["task"])}</td><td>{html.escape(v["title"])}</td>'
            f'<td>{html.escape(v["target"])}</td><td>{html.escape(v["result"])}</td>'
            f'<td><span class="pill {v["status"]}">{label[v["status"]]}</span></td>'
            f'<td><code>{html.escape(v["link"])}</code></td></tr>')
    page = page.replace("{{VERDICT_ROWS}}", "\n".join(rows))
    for key in ("curve", "goal", "horizon", "baseline", "anomaly", "state", "chronos", "edge"):
        page = page.replace("{{CAP_" + key.upper() + "}}", html.escape(cap.get(key, "보고서가 아직 없다.")))
    cond = (d.get("t3") or {}).get("conditions") or []
    page = page.replace("{{CONDITIONS}}", "\n".join(f"<li>{html.escape(x)}</li>" for x in cond))
    page = page.replace("{{GENERATED}}", d["generated_at"])
    page = page.replace("{{DATA}}", json.dumps(d, ensure_ascii=False).replace("</", "<\\/"))
    if fragment:   # 게시용: 문서 뼈대 없이 본문만 (게시 쪽이 뼈대를 붙인다)
        return page
    return ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + page + "</html>\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "summary.html"))
    ap.add_argument("--hours", type=float, default=12, help="과제 8 요약 구간 (logs/service_cycles.jsonl 이 있을 때)")
    ap.add_argument("--fragment", action="store_true", help="<html> 뼈대 없이 본문만 (게시용)")
    args = ap.parse_args()
    d = build(hours=args.hours)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(d, args.fragment), encoding="utf-8")
    print(f"{out}  (과제 판정 {len(d['verdicts'])}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
