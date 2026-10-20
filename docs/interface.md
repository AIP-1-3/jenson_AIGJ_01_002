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
