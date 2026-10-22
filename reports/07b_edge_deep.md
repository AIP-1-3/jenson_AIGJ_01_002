# 과제 7 심화. INT8 · 동시 적재 메모리

## 측정 조건
- 장비: Jetson Orin Nano Super 8GB, JetPack 7.2.1, 전력 모드 MAXN_SUPER(변경 안 함), 클럭 고정 안 함, gpu-run.sh 락 안에서 실행
- 배치 [1, 4], 변환 비교 창 GRU 2048개·오토인코더 8192개 (시험 기간)
- 실행 시간 121s

## 1. INT8 변환

- TensorRT 파이썬 10.16.2.10, trtexec 있음. 보정 데이터: 시험 이전(학습+검증) 기간 창 512개 무작위 (시험 기간은 보정에 쓰지 않음)
- 1순위 TensorRT 파이썬 엔트로피 보정(IInt8EntropyCalibrator2), 실패하면 onnxruntime 정적 양자화 Q/DQ ONNX → trtexec --int8. 두 경우 모두 --fp16 을 함께 켜서 INT8 이 안 되는 층은 FP16 으로 둔다
- TensorRT 지연은 trtexec Latency (입출력 복사 포함, 워밍업 500ms + 200회). onnxruntime 은 반복 200회·워밍업 20회, CPU 스레드 3

### 시도 기록

| model        | path                                    | result                                                             |
|:-------------|:----------------------------------------|:-------------------------------------------------------------------|
| forecast_gru | TensorRT 파이썬 IInt8EntropyCalibrator2 | 엔진 생성 성공                                                     |
| forecast_gru | onnxruntime 정적 양자화 (Q/DQ ONNX)     | 성공                                                               |
| forecast_gru | INT8 엔진 층 정밀도                     | INT8 로 도는 층 0/111 (나머지는 FP16/FP32 — RNN 등 INT8 미지원 층) |
| anomaly_ae   | TensorRT 파이썬 IInt8EntropyCalibrator2 | 엔진 생성 성공                                                     |
| anomaly_ae   | onnxruntime 정적 양자화 (Q/DQ ONNX)     | 성공                                                               |
| anomaly_ae   | INT8 엔진 층 정밀도                     | INT8 로 도는 층 0/9 (나머지는 FP16/FP32 — RNN 등 INT8 미지원 층)   |

### 지연 (ms)

|                                          |   ('mean_ms', 1) |   ('mean_ms', 4) |   ('p95_ms', 1) |   ('p95_ms', 4) |
|:-----------------------------------------|-----------------:|-----------------:|----------------:|----------------:|
| ('anomaly_ae', 'onnxruntime_cpu')        |            0.029 |            0.029 |           0.031 |           0.03  |
| ('anomaly_ae', 'onnxruntime_int8_qdq')   |            0.048 |            0.048 |           0.052 |           0.049 |
| ('anomaly_ae', 'tensorrt_fp16')          |            0.008 |            0.008 |           0.009 |           0.008 |
| ('anomaly_ae', 'tensorrt_int8_calib')    |            0.007 |            0.007 |           0.008 |           0.008 |
| ('forecast_gru', 'onnxruntime_cpu')      |            0.123 |            0.275 |           0.131 |           0.295 |
| ('forecast_gru', 'onnxruntime_int8_qdq') |            0.131 |            0.283 |           0.137 |           0.297 |
| ('forecast_gru', 'tensorrt_fp16')        |            0.008 |            0.011 |           0.012 |           0.02  |
| ('forecast_gru', 'tensorrt_int8_calib')  |            0.009 |            0.01  |           0.014 |           0.016 |

### 예측 GRU: PyTorch CPU 대비 (CO₂ ppm, 환기 판정 = 60분 안 예측 ≥ 1000ppm)

|                      |   max_abs_diff_ppm |   mean_abs_diff_ppm |   vent_flag_change_% |
|:---------------------|-------------------:|--------------------:|---------------------:|
| onnxruntime_cpu      |             0.0001 |              0      |               0      |
| tensorrt_fp16        |             0.4556 |              0.0166 |               0      |
| onnxruntime_int8_qdq |             3.0953 |              1.0509 |               0.0488 |
| tensorrt_int8_calib  |             0.3171 |              0.0166 |               0      |

### 오토인코더: PyTorch CPU 대비 이상 판정 변화율 (%, 8192개 창)

|                      |   flag_change_% |   max_abs_diff_recon |
|:---------------------|----------------:|---------------------:|
| onnxruntime_cpu      |               0 |              0       |
| tensorrt_fp16        |               0 |              0.00574 |
| onnxruntime_int8_qdq |               0 |              0.35571 |
| tensorrt_int8_calib  |               0 |              0.00129 |

### 측정 중 장비 기록 (tegrastats)

|                      |   value |
|:---------------------|--------:|
| samples              |   471   |
| ram_used_max_mb      |  6847   |
| gpu_util_mean_%      |    17.9 |
| power_vdd_in_mean_mw |  7048   |
| power_vdd_in_max_mw  |  8670   |

## 2. 모델 동시 적재 메모리

- 순서대로 한 프로세스에 올림 (각 단계 = 불러오기 + 추론 1회). 시작 RSS 89MB (파이썬·numpy·pandas), 시작 때 보드 여유(MemAvailable) 2735MB → 끝 2487MB
- 상태 분류기: models/state.joblib. ONNX 행은 edge.py 가 만든 models/*.onnx 가 있을 때만
- RSS 는 CPU 쪽 메모리. ONNX·torch 를 둘 다 올린 것은 비교용이며 실제 서비스는 둘 중 하나만 쓴다
- 보드 여유는 공용 장비라 다른 팀원 작업에 따라 바뀐다 (측정 시점 값)

| step                              |   rss_increase_mb |   rss_after_mb |   mem_available_after_mb |   available_drop_mb |   sec | note   |
|:----------------------------------|------------------:|---------------:|-------------------------:|--------------------:|------:|:-------|
| torch import                      |             177.3 |            266 |                     2633 |                 105 |   2.7 |        |
| GRU (torch)                       |              11.3 |            277 |                     2637 |                  -3 |   0.2 |        |
| GRU (ONNX Runtime)                |              27.1 |            305 |                     2631 |                   6 |   0.3 |        |
| 오토인코더 (torch)                |               0.9 |            305 |                     2632 |                  -0 |   0.2 |        |
| 오토인코더 (ONNX Runtime)         |               0.3 |            306 |                     2634 |                  -3 |   0.2 |        |
| 상태 분류기 (sklearn)             |              69.7 |            375 |                     2586 |                  48 |   1.7 |        |
| Chronos-Bolt (chronos-bolt-small) |             313.1 |            689 |                     2487 |                 110 |   8   |        |

- RSS 증가 합계 (모든 단계): 600MB

### 로컬 LLM 과 함께 올리는 조합 (제안)

- LLM 크기는 **가정**이다 (실측 아님): LLM 1B 4bit (가정 ≈1.2GB), LLM 3B 4bit (가정 ≈2.5GB). 4bit 가중치 + KV 캐시·런타임 대략값
- left_mb = 시작 때 보드 여유 − 모델 RSS 합 − LLM. 판정: ≥1500MB 여유, ≥500MB 빠듯, 그 아래 부족 (스왑에 들어가면 지연이 크게 튄다)

| combo                                                       | llm                       |   models_mb |   llm_mb |   left_mb | verdict   | note   |
|:------------------------------------------------------------|:--------------------------|------------:|---------:|----------:|:----------|:-------|
| 서비스 ONNX (GRU·오토인코더 ONNX + 상태 분류기, torch 없이) | LLM 없음                  |          97 |        0 |      2638 | 여유      |        |
| 서비스 ONNX (GRU·오토인코더 ONNX + 상태 분류기, torch 없이) | LLM 1B 4bit (가정 ≈1.2GB) |          97 |     1200 |      1438 | 빠듯      |        |
| 서비스 ONNX (GRU·오토인코더 ONNX + 상태 분류기, torch 없이) | LLM 3B 4bit (가정 ≈2.5GB) |          97 |     2500 |       138 | 부족      |        |
| 서비스 torch (torch + GRU + 오토인코더 + 상태 분류기)       | LLM 없음                  |         259 |        0 |      2476 | 여유      |        |
| 서비스 torch (torch + GRU + 오토인코더 + 상태 분류기)       | LLM 1B 4bit (가정 ≈1.2GB) |         259 |     1200 |      1276 | 빠듯      |        |
| 서비스 torch (torch + GRU + 오토인코더 + 상태 분류기)       | LLM 3B 4bit (가정 ≈2.5GB) |         259 |     2500 |       -24 | 부족      |        |
| 서비스 torch + Chronos-Bolt                                 | LLM 없음                  |         572 |        0 |      2163 | 여유      |        |
| 서비스 torch + Chronos-Bolt                                 | LLM 1B 4bit (가정 ≈1.2GB) |         572 |     1200 |       963 | 빠듯      |        |
| 서비스 torch + Chronos-Bolt                                 | LLM 3B 4bit (가정 ≈2.5GB) |         572 |     2500 |      -337 | 부족      |        |

## 해석 (직접 작성)
- INT8 이 FP16 보다 빨라졌나, 정확도(ppm·판정 변화)는 얼마나 잃었나, GRU 에서 INT8 이 안 된 층과 이유:
- 이 모델 크기에서 INT8 이 의미 있는가 (이미 지연이 0.01ms 수준이면 이득이 거의 없음):
- 로컬 LLM 과 함께 올릴 조합 결론 (어느 모델을 빼거나 ONNX 로 바꿀지):
