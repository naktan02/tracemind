# FedAvg Aggregation

`fedavg/`는 example_count 기반 FedAvg의 공통 산술과 strategy wiring을 소유한다.
adapter family별 delta 해석과 next-state 계산은 각 family package가 소유한다.

## 파일 역할

- `weighted_average.py`: scalar, vector, mapping의 공통 가중 평균 산술
- `strategy.py`: family별 projection을 실행하는 generic FedAvg strategy wiring

이 계층은 `main_server`를 import하지 않는다. model revision, aggregated_at,
publication metadata는 server context가 제공하고, server-owned artifact ref는
main_server가 넘긴 resolver를 통해서만 생성한다.

Adapter family별 FedAvg core와 payload projection은 각 family가 소유한다.

- `methods/adaptation/diagonal_scale/fedavg.py`
- `methods/adaptation/diagonal_scale/fedavg_projection.py`
- `methods/adaptation/classifier_head/fedavg.py`
- `methods/adaptation/classifier_head/fedavg_projection.py`
- `methods/adaptation/lora_classifier/fedavg.py`
- `methods/adaptation/lora_classifier/fedavg_projection.py`
