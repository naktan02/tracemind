# FL Strategy Axis Config

`conf/strategy_axes/fl/`는 FL SSL 비교에서 바꾸는 전략 축을 둔다.
YAML `# @package`는 기존 compose shape를 유지하므로, 폴더명과 compose 후 필드명이
항상 같지는 않다.

## 그룹 의미

| 그룹 | compose field | 의미 |
|---|---|---|
| `method_descriptor/` | `ssl_method` | FedAvg, FedMatch 같은 FL SSL method identity와 report metadata |
| `client_training_profile/` | `training_algorithm_profile` | agent local update를 만들 때 쓰는 training/evidence/scoring/privacy 조합 |
| `shard_policy/` | `shard_policy` | non-IID client split 방식 |

## `client_training_profile` 주의

현재 이름은 `client_training_profile`이지만 compose 후에는
`training_algorithm_profile`로 들어간다. 이 profile은 local training hyperparameter
전체가 아니라 아래 runtime 조합을 한 번에 묶는 profile이다.

- `training_backend_name`
- pseudo-label threshold / acceptance policy
- example generation / evidence / scorer / score policy
- privacy guard
- 현재 기본 adapter family와 aggregation backend 선택값

즉 이 그룹은 pure client-only preset이 아니다. 현재 FedAvg pseudo-label baseline에서는
local update family와 server aggregation backend가 함께 움직이므로 같은 profile에
묶여 있다. future method에서 client step과 server step이 독립적으로 갈라지면
`round_runtime` 또는 별도 group으로 분리한다.

## `method_descriptor`와의 차이

`method_descriptor`는 논문 method의 identity, report role, custom runtime 필요 여부를
표현한다. 실제 local update 계산 조합은 현재 `training_algorithm_profile`에서 온다.

따라서 새 논문 method를 추가할 때는 descriptor config만 추가하지 않는다. 먼저
`methods/federated_ssl/` descriptor와 필요한 `agent`, `main_server`, `methods`
runtime seam을 구현한 뒤 이 config group을 연다.
