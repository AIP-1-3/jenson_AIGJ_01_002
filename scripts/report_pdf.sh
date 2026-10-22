#!/usr/bin/env bash
# docs/report.md → docs/report.pdf (제출용). sudo 불필요.
# 필요: pandoc, 크롬/크로미움(headless), 한글 글꼴(fonts-nanum 권장).
#   ./scripts/report_pdf.sh                 # 기본 브라우저 자동 탐색
#   CHROME=/path/to/chrome ./scripts/report_pdf.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=docs/report.md
OUT=docs/report.pdf
HTML=$(mktemp --suffix=.html -p docs)
trap 'rm -f "$HTML"' EXIT

CHROME=${CHROME:-$(command -v chromium || command -v chromium-browser || command -v google-chrome || true)}
[ -n "$CHROME" ] || { echo "크롬/크로미움을 찾지 못함 (CHROME=경로 로 지정)" >&2; exit 1; }

# 표가 많은 문서라 A4 가로 폭에 맞게 글자를 줄이고, 표는 줄바꿈 허용
cat > "${HTML%.html}.css" <<'CSS'
@page { size: A4; margin: 14mm 12mm; }
body { font-family: "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei", sans-serif; font-size: 9.5pt; line-height: 1.5; max-width: none; margin: 0; }
h1 { font-size: 17pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 13pt; border-bottom: 1px solid #999; margin-top: 18px; page-break-after: avoid; }
h3 { font-size: 11pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; font-size: 8pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; word-break: keep-all; overflow-wrap: anywhere; }
td:first-child, th:first-child { min-width: 4.5em; }
th { background: #f0f0f0; }
code { font-family: "NanumGothicCoding", monospace; font-size: 8.5pt; background: #f5f5f5; padding: 0 2px; }
pre { background: #f5f5f5; padding: 6px; white-space: pre-wrap; font-size: 8pt; }
img { max-width: 100%; }
blockquote { border-left: 3px solid #ccc; margin: 6px 0; padding-left: 8px; color: #444; }
CSS
CSS_FILE="${HTML%.html}.css"
LUA_FILE="${HTML%.html}.lua"
trap 'rm -f "$HTML" "$CSS_FILE" "$LUA_FILE"' EXIT

# PDF 안에서도 근거 링크가 열리도록 저장소 상대 링크를 GitHub 주소로 바꾼다 (그림은 그대로 끼워 넣음)
REPO_URL=${REPO_URL:-https://github.com/AIP-1-3/jenson_AIGJ_01_002/blob/main}
cat > "$LUA_FILE" <<LUA
local base = "$REPO_URL"
local function norm(path)
  local parts = {}
  for seg in ("docs/" .. path):gmatch("[^/]+") do
    if seg == ".." then table.remove(parts) elseif seg ~= "." then table.insert(parts, seg) end
  end
  return table.concat(parts, "/")
end
function Link(el)
  local t = el.target
  if not t:match("^%a[%w+.-]*:") and not t:match("^#") then
    el.target = base .. "/" .. norm(t)
  end
  return el
end
LUA

pandoc "$SRC" -f gfm -t html5 --standalone --embed-resources \
  --resource-path=docs:. --css "$CSS_FILE" --lua-filter "$LUA_FILE" \
  --metadata title="3.2 시계열 추론 보고서" --metadata pagetitle="3.2 시계열 추론 보고서" \
  -V title= -o "$HTML"

"$CHROME" --headless --no-sandbox --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$PWD/$OUT" "file://$PWD/$HTML" 2>/dev/null
echo "저장: $OUT ($(du -h "$OUT" | cut -f1))"
