"""reports/*.md 의 표(pandas to_markdown 형식)를 다시 DataFrame 으로 읽는다.

보고서가 결과의 단일 출처라서, 요약 페이지는 숫자를 따로 저장하지 않고 이 표에서 바로 뽑는다.
그래야 run_all.sh 로 보고서를 다시 만들면 요약 숫자도 같이 바뀌고, 문서끼리 숫자가 어긋나지 않는다.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd


def _cell(s: str):
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):      # ('사무실', 'co2') 같은 튜플 머리글
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
    if s in ("True", "False"):
        return s == "True"
    if s == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return s


def _table(lines: list[str]) -> pd.DataFrame:
    rows = [[c for c in ln.strip().strip("|").split("|")] for ln in lines]
    header = [_cell(c) if c.strip() else "index" for c in rows[0]]
    body = [[_cell(c) for c in r] for r in rows[2:]]   # 1행은 |:---|---:| 구분선
    df = pd.DataFrame(body, columns=header)
    if header[0] == "index":
        df = df.set_index("index")
        df.index.name = None
    return df


def read_tables(path: Path) -> dict[str, list[pd.DataFrame]]:
    """{가장 가까운 제목: [표, ...]}. 제목은 '#' 를 뺀 글자 그대로."""
    out: dict[str, list[pd.DataFrame]] = {}
    heading, buf = "", []
    for ln in path.read_text(encoding="utf-8").splitlines() + [""]:
        if ln.startswith("|"):
            buf.append(ln)
            continue
        if buf:
            if len(buf) >= 2 and re.match(r"^\|[\s:|-]+\|$", buf[1].strip()):
                out.setdefault(heading, []).append(_table(buf))
            buf = []
        if m := re.match(r"^#+\s+(.*)", ln):
            heading = m.group(1).strip()
    return out


def find(tables: dict[str, list[pd.DataFrame]], prefix: str, i: int = 0) -> pd.DataFrame | None:
    """제목이 prefix 로 시작하는 i 번째 표. 없으면 None (보고서가 아직 없을 때 요약을 멈추지 않게)."""
    for title, dfs in tables.items():
        if title.startswith(prefix) and len(dfs) > i:
            return dfs[i]
    return None


def bullets(path: Path, heading_prefix: str) -> list[str]:
    """제목 아래 '- ' 목록 (측정 조건 표시용)."""
    lines, on = [], False
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#"):
            on = ln.lstrip("#").strip().startswith(heading_prefix)
            continue
        if on and ln.startswith("- "):
            lines.append(ln[2:].strip())
    return lines
