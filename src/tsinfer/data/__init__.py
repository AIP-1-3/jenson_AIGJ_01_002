"""과제 2: 정제·리샘플링·시간순 분할·기준선.

학습과 서비스(과제 8)가 같은 함수를 쓰도록 전처리는 여기 한곳에만 둔다.
TODO:
  - preprocess.py : valid_range 제거, 짧은/긴 결측 처리, 5분 리샘플 (configs/default.yaml)
  - split.py      : train/valid/test 시간순 분할, 정규화 통계는 train 으로만
  - baselines.py  : 현재값 유지 / 최근 1시간 평균 / 어제 같은 시각, 5·15·30·60분 MAE
"""
