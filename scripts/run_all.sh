#!/usr/bin/env bash
# 과제 2~7 보고서(와 심화 보고서)를 순서대로 한 번에 만든다 (tmux 안에서 실행 권장).
#   ./scripts/run_all.sh            # 전부
#   ./scripts/run_all.sh 4 5        # 고른 과제만
#   ./scripts/run_all.sh deep       # 심화(B) 전부: 1b 3b 4b 5b 6b 7b 8b
#   ./scripts/run_all.sh 4b 7b      # 고른 심화만
#   ./scripts/run_all.sh 3c         # 과제 3 개선(GRU 평균+Chronos) → reports/03c_forecast_ens.md
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
[[ "${STEPS[*]}" == "deep" ]] && STEPS=(1b 3b 4b 5b 6b 7b 8b)

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
    # 심화(B): 결과는 reports/0Nb_*.md. 기존 보고서·모델 파일은 덮어쓰지 않는다.
    1b) run 1b "저장 용량·재전송" $PY -m tsinfer.collector.storage ;;
    3b) run 3b "LSTM·TCN·lookback·한 공간 빼기" gpu $PY -m tsinfer.models.forecast_deep ;;
    3c) run 3c "GRU 평균+Chronos 섞기(개선)" gpu $PY -m tsinfer.models.forecast_ens ;;
    4b) run 4b "예측오차·영점 틀어짐 탐지" $PY -m tsinfer.models.anomaly_deep ;;
    5b) run 5b "한 공간 빼기·시간표 대조" $PY -m tsinfer.models.state_deep ;;
    6b) run 6b "GRU+Chronos 앙상블·공변량" gpu $PY -m tsinfer.models.pretrained --ensemble ;;
    7b) run 7b "INT8·동시 적재 메모리" gpu $PY -m tsinfer.models.edge_deep ;;
    8b) run 8b "예측형 환기 시뮬레이션" $PY -m tsinfer.service.control_sim ;;
    *) echo "모르는 과제 번호: $s" ;;
  esac
done
echo "보고서: $(ls reports/*.md | tr '\n' ' ')" | tee -a "$LOG"
