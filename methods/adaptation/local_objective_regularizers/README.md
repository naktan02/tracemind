# Local Objective Regularizers

`methods/adaptation/local_objective_regularizers/`는 update/payload family와 분리된
client-local objective regularization 계산을 소유한다.

FedProx처럼 local loss에 추가항을 더하지만 shared update payload shape나 server
aggregation 의미를 바꾸지 않는 계산은 여기에 둔다. 특정 update family의 training
loop는 이 package의 계산을 호출만 하고, regularizer 의미를 소유하지 않는다.
`fedprox.py`는 proximal loss 계산뿐 아니라 round-start snapshot과 적용 조건을 묶는
`prepare_fedprox_regularizer`도 함께 소유한다.

새 regularizer가 server/client round state, update payload, aggregation policy를
요구하면 이 package에 억지로 넣지 않는다. 그런 경우에는 `methods/federated_ssl/`의
round-state capability나 method-owned policy 축을 함께 열어야 한다.

## 금지

- shared adapter state/update payload import
- agent/main_server/scripts runtime import
- FedMatch/FedLGMatch 같은 SSL method-specific policy 소유
- aggregation strategy 또는 server update policy 소유
