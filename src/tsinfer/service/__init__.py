"""과제 8: 상시 추론 서비스.

TODO:
  - 매분 공간별 예측·이상·상태 계산 (data/ 의 전처리 재사용)
  - MQTT 발행은 retain=True (새 구독자도 최신값을 바로 받음)
  - 조회 API (FastAPI) — API_PORT (내 포트 대역 8100~8199)
  - 데이터 끊김 경보, 한 주기 실패해도 다음 주기 정상화
  - 결과 형식 문서: docs/interface.md
"""
