# FedAvg Aggregation

`fedavg/`는 example_count 기반 FedAvg 계산을 소유한다.

## 파일 역할

- `fedavg.py`: scalar, vector, mapping의 공통 가중 평균 산술
- `diagonal_scale_fedavg.py`: diagonal-scale delta를 평균해 next dimension scale을 계산
- `classifier_head_fedavg.py`: classifier-head weight/bias delta를 평균해 next head 값을 계산
- `lora_classifier_fedavg.py`: LoRA parameter delta와 classifier-head delta를 함께 평균

이 계층은 `main_server`를 import하지 않는다. model revision, aggregated_at,
publication metadata는 server adapter가 붙인다.
