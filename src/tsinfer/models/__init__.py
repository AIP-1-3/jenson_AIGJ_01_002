"""과제 3~6: 모델.

TODO:
  - forecast.py  : 60분(5분×12) 예측 — LSTM/GRU/TCN, 일정 포함/제외 비교
  - anomaly.py   : 오토인코더 + 규칙(급변·값 고정), 원인 센서 표시
  - state.py     : 사용 여부(일정은 입력 금지) / 30분 내 1,000ppm 환기 선제 경보
  - pretrained.py: Chronos 계열 zero-shot, 예측 구간 적중률·지연·메모리
  - export.py    : ONNX → TensorRT FP16 (과제 7, 빌드는 gpu-run.sh 로)
"""
