# FedAvg Aggregation

`fedavg/`는 example_count 기반 FedAvg 계산을 소유한다.

## 파일 역할

- `fedavg.py`: scalar, vector, mapping의 공통 가중 평균 산술
- `diagonal_scale_fedavg.py`: diagonal-scale delta를 평균해 next dimension scale을 계산
- `classifier_head_fedavg.py`: classifier-head weight/bias delta를 평균해 next head 값을 계산
- `lora_classifier_fedavg.py`: LoRA parameter delta와 classifier-head delta를 함께 평균
- `strategy.py`: family별 projection을 실행하는 generic FedAvg strategy wiring

이 계층은 `main_server`를 import하지 않는다. model revision, aggregated_at,
publication metadata는 server context가 제공하고, server-owned artifact ref는
main_server가 넘긴 resolver를 통해서만 생성한다.

Adapter family별 payload projection은 각 family가 소유한다.

- `methods/adaptation/diagonal_scale/fedavg_projection.py`
- `methods/adaptation/classifier_head/fedavg_projection.py`
- `methods/adaptation/lora_classifier/fedavg_projection.py`
