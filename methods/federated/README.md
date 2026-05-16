# Federated Methods

`methods/federated/`는 federated learning에서 재사용되는 순수 method 계산을 둔다.

현재 `aggregation/fedavg/`는 example_count 기반 FedAvg의 공통 가중 평균 산술과
strategy wiring을 소유한다. adapter family별 delta 해석과 next-state 계산은
`methods/adaptation/<family>/aggregation/fedavg.py`가 소유한다.

`shard_policy/`는 FL SSL non-IID 비교에서 재사용되는 label-dominant와
Dirichlet label-skew client assignment 계산을 소유한다.

반면 round 생성, client update acceptance, model revision 발행,
artifact publication은 `main_server` 책임으로 유지한다.
