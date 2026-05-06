# Federated Methods

`methods/federated/`는 federated learning에서 재사용되는 순수 method 계산을 둔다.

현재 `aggregation/fedavg/`가 example_count 기반 FedAvg의 가중 평균과
adapter family별 next-state 값 계산을 소유한다. `lora_classifier`는 LoRA
parameter delta와 classifier-head delta를 같은 method core에서 함께 평균한다.

`shard_policy/`는 FL SSL non-IID 비교에서 재사용되는 label-dominant와
Dirichlet label-skew client assignment 계산을 소유한다.

반면 round 생성, client update acceptance, model revision 발행,
artifact publication은 `main_server` 책임으로 유지한다.
