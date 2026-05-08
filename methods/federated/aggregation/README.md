# Federated Aggregation Methods

`methods/federated/aggregation/`는 server runtime과 분리된 federated aggregation
계산 core를 둔다.

## 책임

- FedAvg 같은 method-level 가중 평균 산술
- adapter family별 next-state 값 계산
- 빈 update, total weight 0, key/dimension mismatch 같은 method-level validation
- aggregation method strategy wiring
- method metadata registration은 core function 옆 decorator가 소유하고,
  `builtin_loader.py`는 builtin method module import만 담당

## 제외

- round lifecycle, update acceptance, publication metadata
- `TrainingUpdateEnvelope`, `AggregationResult` 같은 main_server boundary model
- catalog/API 노출

Adapter family별 payload projection은 `methods/adaptation/<family>/`가 소유한다.
`main_server`는 round lifecycle, update storage, server-owned artifact ref 생성,
publication metadata만 맡고, selected aggregation strategy를 호출한다.
