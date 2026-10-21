#!/usr/bin/env bash
# 수집기 중단 시험 (과제 1·8 동작 근거). sudo 불필요 — 사용자 서비스만 멈췄다 다시 켠다.
#   ./scripts/outage_test.sh [분, 기본 10]
# 멈춘 시각·다시 켠 시각을 logs/outage_test.log 에 남긴다.
# 확인: 다음 날 `python -m tsinfer.collector.coverage --hours 24` 의 gaps_all 에 같은 구간이 보이고,
#       `python -m tsinfer.service.summary --hours 12` 의 data_gap_minutes 에 끊김 경보가 잡혀야 한다.
set -euo pipefail
cd "$(dirname "$0")/.."
MIN="${1:-10}"
LOG=logs/outage_test.log
mkdir -p logs
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "$(stamp) stop tsinfer-collector (${MIN}분)" | tee -a "$LOG"
systemctl --user stop tsinfer-collector
trap 'systemctl --user start tsinfer-collector; echo "$(stamp) start tsinfer-collector (중간 종료)" | tee -a "$LOG"' INT TERM
sleep "$((MIN * 60))"
systemctl --user start tsinfer-collector
trap - INT TERM
echo "$(stamp) start tsinfer-collector" | tee -a "$LOG"
systemctl --user is-active tsinfer-collector | sed "s/^/$(stamp) 상태: /" | tee -a "$LOG"
