# 시계열 데이터 추론 — 개인 저장소 (AIP 3.2)

학습 공간(강의실 2, 회의실 1, 라운지 1)의 CO₂·온도·습도로 60분 예측, 이상 탐지, 사용 여부·환기 경보를 내고
팀 Jetson(Orin Nano Super 8GB)에서 상시 서비스로 운영한다. 과제 원문: `AIP-1/timeseries-inference`.

## 빠른 시작 (Jetson)

```bash
git clone https://github.com/AIP-1-3/jenson_AIGJ_01_002.git ~/jenson_AIGJ_01_002
cd ~/jenson_AIGJ_01_002
python3 -m venv .venv --system-site-packages   # JetPack 의 TensorRT 바인딩 재사용
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .

cp .env.example .env && chmod 600 .env          # 접속 정보는 .env 에만
.venv/bin/python -m pytest

# 수집기 시험 실행 (tmux 안에서)
.venv/bin/python -m tsinfer.collector
```

과거 데이터(InfluxDB `iot-bucket`, 2026-07-22~)는 교육장망에서 parquet 로 받는다. 끊겨도 같은 명령으로 이어 받는다.

```bash
.venv/bin/python -m tsinfer.ingest.influx_export --start 2026-07-01
```

과제 2 기준선 성능표 (시험 기간, `reports/02_baselines.md` 로 저장):

```bash
.venv/bin/python -m tsinfer.data.baselines
.venv/bin/python -m tsinfer.data.diagnose     # 진단서 reports/02_diagnosis.md + 그림
```

과제 3 GRU 예측 (PyTorch CPU 로 충분한 크기. 설치는 아래 한 번):

```bash
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # NumPy 를 건드리지 않는 CPU 빌드
tmux new -s train
.venv/bin/python -m tsinfer.models.forecast --compare-time   # → models/forecast_gru.pt, reports/03_forecast.md
```

과제 4 이상 탐지 (고장 5종을 시험 기간에 넣어 규칙/오토인코더/결합 채점):

```bash
.venv/bin/python -m tsinfer.models.anomaly    # → models/anomaly_ae.pt, reports/04_anomaly.md
```

과제 5 상태 추정 (사용 여부 정답은 `configs/schedule.yaml` 의 시간표, 직접 관찰은 `data/observations.csv`):

```bash
.venv/bin/python -m tsinfer.models.state      # → models/state.joblib, reports/05_state.md
```

과제 6 사전학습 모델 비교 (Chronos-Bolt zero-shot, 첫 실행 때 모델을 내려받는다):

```bash
.venv/bin/pip install chronos-forecasting "numpy<2"
~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.pretrained   # → reports/06_pretrained.md
```

과제 7 엣지 최적화 (과제 3·4 모델 파일이 있어야 한다):

```bash
.venv/bin/pip install onnx onnxruntime "numpy<2"
~/jetson_orin/scripts/gpu-run.sh -- .venv/bin/python -m tsinfer.models.edge         # → reports/07_edge.md
```

학습이 끝나면 `.env` 의 `SERVICE_MODEL=gru` 로 바꾸고 `systemctl --user restart tsinfer-service`.

상시 추론 서비스 (과제 8, 형식은 [docs/interface.md](docs/interface.md)):

```bash
.venv/bin/python -m tsinfer.service --once        # 한 주기만 돌려 결과 확인
.venv/bin/python -m tsinfer.service --no-publish  # API 만 (http://localhost:8100/docs)
.venv/bin/python -m tsinfer.service.summary       # 최근 12시간 무중단 보고
```

상시 가동은 [tsinfer-collector.service](deploy/systemd/tsinfer-collector.service), [tsinfer-service.service](deploy/systemd/tsinfer-service.service) 주석대로 user 서비스로 등록한다.

## 구조

```
configs/
  default.yaml        공간·전처리·분할·예측 지평·경보 임계값 (학습과 서비스가 같이 읽음)
  sensors.yaml        MQTT 토픽·페이로드 매핑 (형식이 바뀌면 이 파일만 수정)
src/tsinfer/
  config.py           .env + yaml 로딩
  collector/          과제 1: 상시 수집기 (자동 재접속, SQLite WAL, 재전송 중복 제거)
  ingest/             InfluxDB 과거 데이터 → data/raw/*.parquet
  data/               과제 2: 정제·리샘플·분할·기준선
  models/             과제 3~7: 예측·이상·상태·사전학습 비교·TensorRT 변환
  service/            과제 8: 매분 추론, MQTT(retain) 발행, 조회 API
deploy/systemd/       재부팅 후 자동 시작
docs/progress.md      과제별 확인 기준 체크리스트
reports/              과제별 결과 (측정 조건 반드시 기재)
notebooks/            EDA (출력 지우고 커밋)
tests/
```

## 공용 Jetson 규칙 (3명 공유)

- 내 포트 대역: **8100~8199** (API 8100). 공용 데모 8000번대, minseo 8200번대, seongchan 8300번대.
- 상시 프로세스는 메모리 상한을 건다 (수집기 300MB, 추론 서비스 1.5GB).
- 학습·TensorRT 빌드·사전학습 모델 실행은 `~/jetson_orin/scripts/gpu-run.sh -- <명령>` 으로 순서 대기.
- `MQTT_CLIENT_ID` 는 사람마다 다르게.
- 재부팅·전력 모드·클럭 변경은 팀 합의 후에만 (재부팅 시험은 팀원과 날짜를 맞춰 한 번에).

## 결과 요약

| 과제 | 확인 기준 | 결과 | 보고서 |
|---|---|---|---|
| 1 수집 | 24h 커버율 ≥95%, 브로커 재시작 후 1분 내 복구, 재부팅 후 자동 시작 | – | – |
| 2 기준선 | 3개 기준선 5·15·30·60분 MAE | – | – |
| 3 예측 | 60분 CO₂ MAE 현재값 유지 대비 −30% | – | – |
| 4 이상 | 탐지율 ≥80%, 헛경보 ≤1회/일 | – | – |
| 5 상태 | 환기 경보 재현율 ≥85%, 규칙보다 macro-F1↑ | – | – |
| 6 사전학습 | 정확도·구간 적중률·지연·메모리 비교 | – | – |
| 7 엣지 | 변환 후 CO₂ 차이 ≤1ppm, 분류 변화 ≤0.1% | – | – |
| 8 서비스 | 12시간 무중단, 수집 10분 중단 시 끊김 경보 | – | – |
