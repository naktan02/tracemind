# Federated Aggregation Methods

`methods/federated/aggregation/`는 server runtime과 분리된 재사용 federated
aggregation backend의 generic 산술과 strategy wiring을 둔다.

## 책임

- FedAvg, partitioned FedAvg 같은 reusable backend 가중 평균 산술
- 빈 update, total weight 0, key/dimension mismatch 같은 backend-level validation
- aggregation backend strategy wiring
- method metadata registration은 adapter family projection registration이 제공하고,
  registry는 method name과 adapter kind convention으로 필요한 module만 import

## 제외

- round lifecycle, update acceptance, publication metadata
- `TrainingUpdateEnvelope`, `AggregationResult` 같은 main_server boundary model
- catalog/API 노출

Adapter family별 FedAvg core, delta 해석, payload projection은
`methods/adaptation/<family>/`가 소유한다.
`partitioned_fedavg`는 partition 이름 의미를 소유하지 않는다. adapter family가
partition payload를 materialize하고, FedMatch의 `sigma/psi` 같은 scheme 의미는
`methods/federated_ssl/<method>/`가 소유한다.
특정 논문 method에만 종속된 aggregation 변형은 `methods/federated_ssl/<method>/`에
둘 수 있고, 두 개 이상 method가 공유할 때 이 패키지로 승격한다.
`main_server`는 round lifecycle, update storage, server-owned artifact ref 생성,
publication metadata만 맡고, selected aggregation strategy를 호출한다.
