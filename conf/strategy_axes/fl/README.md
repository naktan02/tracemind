# FL Strategy Axis Config

`conf/strategy_axes/fl/`는 FL SSL 비교에서 바꾸는 전략 축을 둔다.
YAML `# @package`는 기존 compose shape를 유지하므로, 폴더명과 compose 후 필드명이
항상 같지는 않다.

## 그룹 의미

| 그룹 | compose field | 의미 |
|---|---|---|
| `method_descriptor/` | `ssl_method` | FedAvg, FedMatch 같은 FL SSL method identity와 report metadata |
| `local_update_profile/` | `local_update_profile` | agent local update를 만들 때 쓰는 training/evidence/scoring/privacy 조합 |
| `round_runtime_profile/` | `round_runtime_profile` | server round의 adapter family와 aggregation backend 조합 |
| `shard_policy/` | `shard_policy` | non-IID client split 방식 |

## `local_update_profile`와 `round_runtime_profile`

`local_update_profile`은 local training hyperparameter 전체가 아니라 agent가
pseudo-label update를 만들 때 쓰는 아래 조합을 묶는 profile이다.

- `training_backend_name`
- pseudo-label threshold / acceptance policy
- example generation / evidence / scorer / score policy
- privacy guard

`round_runtime_profile`은 server round가 공유할 `adapter_family_name`과
`aggregation_backend_name`을 소유한다. 따라서 같은 local update profile을 유지한 채
다른 aggregation backend를 붙이거나, 같은 FedAvg runtime에 다른 local objective를
붙이는 조합을 config에서 분리해 표현할 수 있다.

## `method_descriptor`와의 차이

`method_descriptor`는 논문 method의 identity, report role, custom runtime 필요 여부를
표현한다. 실제 local update 계산 조합은 `local_update_profile`, server round runtime
조합은 `round_runtime_profile`에서 온다.

따라서 새 논문 method를 추가할 때는 descriptor config만 추가하지 않는다. 먼저
`methods/federated_ssl/` descriptor와 필요한 `agent`, `main_server`, `methods`
runtime seam을 구현한 뒤 이 config group을 연다.
