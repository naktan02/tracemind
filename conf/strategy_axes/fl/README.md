# FL Strategy Axis Config

`conf/strategy_axes/fl/`는 FL SSL 비교에서 바꾸는 전략 축을 둔다.
YAML `# @package`는 기존 compose shape를 유지하므로, 폴더명과 compose 후 필드명이
항상 같지는 않다.

## 그룹 의미

| 그룹 | compose field | 의미 |
|---|---|---|
| `method_descriptor/` | `ssl_method` | FedMatch, FedLGMatch 같은 FL SSL method identity와 report metadata. `fedavg_pseudo_label`은 현재 baseline identity다. |
| `local_update_profile/` | `local_update_profile` | agent local update를 만들 때 쓰는 training/evidence/scoring/privacy 조합 |
| `round_runtime_profile/` | `round_runtime_profile` | server round의 adapter family와 aggregation backend 조합 |
| `experiment_profile/` | `fl_profile` | method/local update/round runtime 축을 함께 고르는 compose preset |
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
`methods/federated_ssl/<method>/` descriptor/recipe와 필요한 methods core를 구현한
뒤 이 config group을 연다. method-only local/server/aggregation 변형은 method 폴더에
둘 수 있고, `agent`/`main_server`에는 capability adapter만 둔다.

## `experiment_profile`

`experiment_profile`은 실행값의 source of truth가 아니라 compose preset이다. 예를 들어
`fedavg_pseudo_label_lora_classifier_v1`은 `method_descriptor`,
`local_update_profile`, `round_runtime_profile`을 함께 고르는 좁은 시작점이다.
실제 threshold, LoRA rank, round 수 같은 실행 파라미터는 계속 Hydra config leaf에
남긴다.

사람이 읽는 method 조립표는 `methods/federated_ssl/<method>/recipe.py`가 소유하고,
Hydra YAML은 실행 조합과 파라미터 값만 소유한다.
