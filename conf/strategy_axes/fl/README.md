# FL Strategy Axis Config

`conf/strategy_axes/fl/`는 FL SSL 비교에서 바꾸는 전략 축을 둔다.
YAML `# @package`는 기존 compose shape를 유지하므로, 폴더명과 compose 후 필드명이
항상 같지는 않다.

## 그룹 의미

| 그룹 | compose field | 의미 |
|---|---|---|
| `method_descriptor/` | `ssl_method` | FedMatch, FedLGMatch 같은 FL SSL method identity와 report metadata. `fedavg_pseudo_label`은 현재 baseline identity다. |
| `local_update_profile/` | `local_update_profile` | agent local update를 만들 때 쓰는 training/evidence/scoring/privacy 조합 |
| `round_runtime.*` | `round_runtime` | server round의 adapter family와 aggregation backend 직접 leaf |
| `shard_policy/` | `shard_policy` | non-IID client split 방식 |

## `fl_method` 실행 계획

FL entrypoint는 `fl_method` section을 `FederatedSslExecutionPlan`으로 해석한다.
이 plan은 사람이 고른 상위 method와 runtime capability를 bootstrap 전에 검증하고,
report protocol에 남긴다.

- `composition_mode=method_owned`: FedMatch/FedLGMatch 같은 상위 method가 client
  objective, server policy, round state exchange 요구사항을 소유한다. 이 모드에서는
  lower axes를 따로 쓰지 않는다.
- `composition_mode=manual`: 논문 method가 아니라 `client_ssl_objective`,
  `server_aggregation`, `update_family`를 직접 조합하는 baseline/ablation 모드다.
  사용자는 `local_update_profile`과 `round_runtime.adapter_family_name` /
  `round_runtime.aggregation_backend_name`을 직접 고른다. lower axes는 compose된
  `ssl_method`와 `round_runtime.*` leaf에서 실행 계획 builder가 자동 파생한다.

`security_policy`는 method가 아니라 runtime capability 축이다. 현재 simulation은
`plaintext`만 지원하고, secure aggregation/DP/암호화 artifact ref는 이후 capability
adapter와 compatibility validator를 붙이는 방식으로 연다.

## `local_update_profile`와 `round_runtime.*`

`local_update_profile`은 local training hyperparameter 전체가 아니라 agent가
pseudo-label update를 만들 때 쓰는 아래 조합을 묶는 profile이다.

- `training_backend_name`
- pseudo-label threshold / acceptance policy
- example generation / evidence / scorer / score policy
- privacy guard

server round 조합은 별도 YAML group이 아니라 최종 compose된
`round_runtime.adapter_family_name`과 `round_runtime.aggregation_backend_name` leaf가
직접 소유한다. baseline/ablation은 profile 파일을 새로 만들지 말고
`round_runtime.*` leaf를 직접 override한다.

## `method_descriptor`와의 차이

`method_descriptor`는 논문 method의 identity, report role, custom runtime 필요 여부를
표현한다. 실제 local update 계산 조합은 `local_update_profile`, server round runtime
조합은 `round_runtime.*` leaf에서 온다.

따라서 새 논문 method를 추가할 때는 descriptor config만 추가하지 않는다. 먼저
`methods/federated_ssl/<method>/` descriptor/recipe metadata와 필요한 methods core를 구현한
뒤 이 config group을 연다. method-only local/server/aggregation 변형은 method 폴더에
둘 수 있고, `agent`/`main_server`에는 capability adapter만 둔다.

사람이 읽는 method 조립표는 `methods/federated_ssl/<method>/descriptor.py`의 recipe
metadata가 소유하고, 커질 때만 optional `recipe.py`로 분리한다. Hydra YAML은 실행
조합과 파라미터 값만 소유한다.

## Compose Preset 제거 기준

method/local update/server round를 한꺼번에 고르는 high-level compose preset은 중복
source-of-truth를 만들기 때문에 두지 않는다. 논문 method 실행은 기본
`fl_method.composition_mode=method_owned`에서 `method_descriptor`와 method-owned
recipe 검증으로 닫고, lower-axis baseline/ablation은 `composition_mode=manual`에서
`local_update_profile`과 `round_runtime.*` leaf를 직접 override한다.
