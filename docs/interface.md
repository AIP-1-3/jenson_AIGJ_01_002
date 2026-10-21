# 서비스 인터페이스 (과제 8) — schema_version 1.0

서비스는 매분(정분 +5초) 한 번 추론하고, 결과를 **조회 API** 와 **MQTT(retain)** 두 곳으로 낸다.
모델을 바꿔도 이 형식은 그대로 둔다. 바뀌면 `schema_version` 을 올린다.

## 입력
- 수집기 SQLite(`DB_PATH`) 최근 6시간 → 학습과 같은 `preprocess.clean` / `to_wide` (5분 칸)
- 막 시작된 5분 칸은 값이 흔들려서 버리고 **완성된 칸까지만** 쓴다.
- 공간의 마지막 수신이 10분보다 오래되면 `data_ok=false`, 예측을 내지 않고 `data_gap` 경보.

## MQTT
- 토픽: `{PUBLISH_PREFIX}/{공간}` (예: `tsinfer/mskim/실습실`), QoS 1, retain
- `.env` 의 `PUBLISH_PREFIX` 가 비어 있으면 발행하지 않는다. 발행 실패는 추론을 멈추지 않는다.
- 접속 ID 는 `MQTT_CLIENT_ID` + `-svc` (수집기와 겹치면 브로커가 한쪽을 끊는다).

## API (`API_PORT`, 기본 8100)

> 호출 위치: Jetson 안(ssh 접속 후)에서는 `curl -s localhost:8100/latest`. 다른 컴퓨터에서는 `localhost` 대신 Jetson 주소(팀 Tailscale 이름 등)를 쓴다. 서비스는 모든 인터페이스(0.0.0.0)에서 받는다.
| 경로 | 설명 |
|---|---|
| `GET /health` | `ok`, 마지막 추론 시각, 연속 실패 수, 공간별 `data_ok` |
| `GET /latest` | 전체 공간 결과 (아래 형식) |
| `GET /latest/{공간}` | 한 공간 결과 (MQTT 페이로드와 같음) |
| `GET /docs` | 자동 문서 (FastAPI) |

## 공간 결과 형식
```json
{
  "schema_version": "1.0",
  "generated_at": "2026-10-20T12:03:05+00:00",
  "model": "trend_baseline_v0",
  "space": "실습실",
  "data_ok": true,
  "last_received": "2026-10-20T12:02:41+00:00",
  "current": {"co2": 812.4, "temperature": 23.1, "humidity": 41.2},
  "forecast": {
    "co2": {"5": 820.1, "10": 827.5, "...": 0, "60": 905.3},
    "temperature": {"5": 23.1, "...": 0, "60": 23.4}
  },
  "co2_ventilate_in_minutes": null,
  "anomaly": {
    "is_anomaly": false,
    "alerts": [{"type": "spike", "metric": "co2", "severity": "medium", "detail": "5분 사이 +320.0"}]
  },
  "state": {"occupied": true, "ventilation_alert": false},
  "horizons_reported": [5, 15, 30, 60]
}
```

| 필드 | 뜻 |
|---|---|
| `forecast.*` | 5~60분 5분 간격 예측. 키는 분(문자열) |
| `co2_ventilate_in_minutes` | 예측 CO₂ 가 `co2_ventilate_ppm`(1,000) 에 처음 닿는 시점. 없으면 null |
| `state.ventilation_alert` | 지금은 1,000 미만인데 30분 안에 넘을 것으로 예측되면 true (이미 넘은 경우 제외) |
| `state.occupied` | v0 는 단순 규칙(15분 CO₂ 상승 ≥30ppm). 과제 5 분류기로 교체 예정. 데이터 부족 시 null |
| `anomaly.alerts[].type` | `data_gap`, `stuck`(값 고정: CO₂ 60분·온습도 180분), `out_of_range`, `spike`, `co2_critical` |
| `anomaly.is_anomaly` | severity=high 경보가 하나라도 있으면 true |

## 운영 기록
- 매 주기를 `logs/service_cycles.jsonl` 에 한 줄씩 남긴다 (성공 여부, 소요 시간, 공간별 data_ok·경보, 발행 여부).
- 12시간 무중단 보고: `python -m tsinfer.service.summary --hours 12`
  → 주기 수/기대 주기 수, 실패 주기, 최대 공백(분), p95 소요 시간, 공간별 끊김 분.

## 자기 채점 (선택, 과제 8 심화)
`.env` 에 `SELF_MONITOR=1` 일 때만 켜진다. 없으면 위 형식·동작 그대로.
- 매 주기 공간별 CO₂ 60분 예측을 보관했다가, 그 5분 칸이 완성되면 엔진의 `current.co2` 와 비교해 채점한다.
- `GET /health` 에 `monitor` 필드가 생긴다 (켜졌을 때만). `logs/service_cycles.jsonl` 각 줄에도 `monitor`(+ 이번 주기 알림 수 `sent`).

| 필드 | 뜻 |
|---|---|
| `monitor.spaces.{공간}.n` / `mae` / `mae_persistence` / `ratio` | 최근 60개 채점의 MAE, 같은 시점 현재값 유지 MAE, 그 비율 |
| `monitor.spaces.{공간}.alerting` | 경고 상태 (MAE ≥150ppm, 또는 비율 >1.2 이고 MAE ≥30ppm; 12개 이상 채점 후) |
| `monitor.pending` / `expired` / `alerts_sent` / `notify` | 채점 대기 예측 수, 데이터 끊김으로 못 채점한 수, 보낸 알림 수, 알림 채널 유무 |

- 알림: `DISCORD_WEBHOOK_URL`(디스코드 웹훅)로 보낸다. 비어 있으면 채점만 하고 알림은 건너뛴다.
  같은 공간 경고는 1시간에 한 번만, 정상으로 돌아오면 한 번 알린다. 전송 실패(3초 타임아웃)는 서비스에 영향 없음.
- `python -m tsinfer.service.summary` 는 기록에 `monitor` 가 있으면 공간별 최근 MAE 와 알림 횟수를 함께 낸다.
