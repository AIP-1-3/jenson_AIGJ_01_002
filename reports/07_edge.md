# 과제 7. 엣지 최적화

## 측정 조건
- 장비: Jetson Orin Nano Super 8GB, JetPack 7.2.1, 전력 모드 MAXN_SUPER(변경 안 함), 클럭 고정 안 함, gpu-run.sh 락 안에서 단독 측정
- torch 2.14.1+cpu (CUDA 없음), CPU 스레드 3, onnxruntime CPU, TensorRT FP16 (trtexec)
- 반복 200회, 워밍업 20회 (TensorRT 는 워밍업 500ms + 200회), 배치 [1, 4]
- 입력: 실데이터 시험 기간 창. TensorRT 지연은 입출력 복사 포함(trtexec Latency)

## 지연 (ms)

|                                     |   ('mean_ms', 1) |   ('mean_ms', 4) |   ('p95_ms', 1) |   ('p95_ms', 4) |
|:------------------------------------|-----------------:|-----------------:|----------------:|----------------:|
| ('anomaly_ae', 'onnxruntime_cpu')   |            0.026 |            0.027 |           0.028 |           0.028 |
| ('anomaly_ae', 'pytorch_cpu')       |            0.894 |            0.859 |           1.236 |           1.421 |
| ('anomaly_ae', 'pytorch_gpu')       |          nan     |          nan     |         nan     |         nan     |
| ('anomaly_ae', 'tensorrt_fp16')     |            0.007 |            0.008 |           0.008 |           0.009 |
| ('forecast_gru', 'onnxruntime_cpu') |            0.119 |            0.266 |           0.129 |           0.285 |
| ('forecast_gru', 'pytorch_cpu')     |           10.977 |           10.568 |          29.615 |          19.243 |
| ('forecast_gru', 'pytorch_gpu')     |          nan     |          nan     |         nan     |         nan     |
| ('forecast_gru', 'tensorrt_fp16')   |            0.01  |            0.009 |           0.012 |           0.011 |

## 변환 전후 차이

### 예측 GRU: PyTorch CPU 대비 CO₂ 예측 최대 차이 (ppm, 목표 ≤1)

|                 |   max_abs_diff_ppm | pass(≤1ppm)   |
|:----------------|-------------------:|:--------------|
| onnxruntime_cpu |             0.0001 | True          |
| tensorrt_fp16   |             0.3171 | True          |

### 오토인코더: 이상 판정이 바뀐 비율 (%, 8192개 창, 목표 ≤0.1%)

|                 |   flag_change_% | pass(≤0.1%)   |
|:----------------|----------------:|:--------------|
| onnxruntime_cpu |               0 | True          |
| tensorrt_fp16   |               0 | True          |

## 측정 중 장비 기록 (tegrastats)

|                      |   value |
|:---------------------|--------:|
| samples              |   214   |
| ram_used_max_mb      |  5379   |
| gpu_util_mean_%      |     3.4 |
| power_vdd_in_mean_mw |  7412   |
| power_vdd_in_max_mw  |  8748   |

## 해석 (직접 작성)
- 배치 1 에서 GPU 가 CPU 보다 느리거나 비슷하다면 그 이유 (작은 모델, 복사 비용):
- 서비스(매분 1회, 공간 3개)에 실제로 필요한 속도와 비교한 결론:
- FP16 오차가 경보 판단에 주는 영향:
