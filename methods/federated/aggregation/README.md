# Federated Aggregation Methods

`methods/federated/aggregation/`는 server runtime과 분리된 federated aggregation
계산 core를 둔다.

## 책임

- FedAvg 같은 method-level 가중 평균 산술
- adapter family별 next-state 값 계산
- 빈 update, total weight 0, key/dimension mismatch 같은 method-level validation

## 제외

- round lifecycle, update acceptance, publication metadata
- `TrainingUpdateEnvelope`, `AggregationResult` 같은 main_server boundary model
- catalog/API 노출과 server runtime override 해석

`main_server`는 accepted update를 이 계층의 method input으로 변환하고, 결과를
server-owned `AggregationResult`와 publication metadata로 조립한다.
