#!/usr/bin/env bash
# 과제 2~7 보고서를 순서대로 한 번에 만든다 (tmux 안에서 실행 권장).
#   ./scripts/run_all.sh            # 전부
#   ./scripts/run_all.sh 4 5        # 고른 과제만
# 무거운 단계(6·7)는 팀 GPU 락(gpu-run.sh) 안에서 돈다. 로그: logs/run_all_<시각>.log
# 진행 보기: tail -f "$(ls -t logs/run_all_*.log | head -1)"
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1   # 에폭 진행이 로그에 바로 찍히게 (파일로 보내면 print 가 버퍼에 쌓임)
GPU_RUN="$HOME/jetson_orin/scripts/gpu-run.sh"
[[ -x "$GPU_RUN" ]] || GPU_RUN=""
mkdir -p logs reports
LOG="logs/run_all_$(date +%Y%m%d_%H%M).log"
STEPS=("$@"); [[ ${#STEPS[@]} -eq 0 ]] && STEPS=(2 3 4 5 6 7)

run() {   # run <과제> <설명> <명령...>
  local n=$1 desc=$2; shift 2
  echo "=== 과제 $n: $desc ($(date +%H:%M:%S)) ===" | tee -a "$LOG"
  free -h | sed -n 2p | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then echo "  ✓ 완료" | tee -a "$LOG"; else echo "  ✗ 실패 — $LOG 확인" | tee -a "$LOG"; fi
}
gpu() { if [[ -n "$GPU_RUN" ]]; then "$GPU_RUN" -- "$@"; else "$@"; fi; }

for s in "${STEPS[@]}"; do
  case $s in
    2) run 2 "기준선·진단서" bash -c "$PY -m tsinfer.data.baselines && $PY -m tsinfer.data.diagnose" ;;
    3) run 3 "GRU 예측" $PY -m tsinfer.models.forecast --compare-time --schedule ;;
    4) run 4 "이상 탐지" $PY -m tsinfer.models.anomaly ;;
    5) run 5 "상태 추정" $PY -m tsinfer.models.state ;;
    6) run 6 "Chronos 비교" gpu $PY -m tsinfer.models.pretrained ;;
    7) run 7 "엣지 최적화" gpu $PY -m tsinfer.models.edge ;;
    *) echo "모르는 과제 번호: $s" ;;
  esac
done
echo "보고서: $(ls reports/*.md | tr '\n' ' ')" | tee -a "$LOG"
