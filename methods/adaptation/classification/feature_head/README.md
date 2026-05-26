# Feature Head Classification

`feature_head/`는 고정 feature 또는 embedding 위의 classifier head adaptation
variant를 소유한다. 이 경로는 text 전용이 아니다.

## 파일

- `bootstrap.py`: prototype centroid 기반 classifier-head 초기 state 생성
- `scoring.py`: shared classifier-head state 기반 logits scoring backend

FedAvg projection과 next-state materialization은
`methods/adaptation/classification/aggregation/feature_head_fedavg_projection.py`가
소유한다.
