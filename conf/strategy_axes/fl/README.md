# FL Strategy Axis Config

`conf/strategy_axes/fl/`는 FL SSL 비교에서 바꾸는 전략 축을 둔다.
YAML `# @package`는 기존 compose shape를 유지하므로, 폴더명과 compose 후 필드명이
항상 같지는 않다.

## 그룹 의미

| 그룹 | compose field | 의미 |
|---|---|---|
| `method_descriptor/` | `ssl_method` | FedMatch, FedLGMatch 같은 method-owned FL SSL method identity, 원본 parameter snapshot, report metadata. 기본 manual baseline은 이 그룹을 compose하지 않는다. |
| `local_update_profile/` | `local_update_profile` | agent local update를 만들 때 쓰는 training/evidence/scoring/privacy 조합 |
| `round_runtime.*` | `round_runtime` | server round의 adapter family와 aggregation backend 직접 leaf |
| `shard_policy/` | `shard_policy` | non-IID client split 방식 |
| `labeled_exposure_policy/` | `labeled_exposure_policy` | 선택된 labeled seed가 client-local split인지, 모든 client가 공유하는 public seed인지, 또는 server-only seed인지 구분 |
| `client_participation_policy/` | `client_participation_policy` | round별 학습 참여 client subset 선택 방식 |
| `local_supervision_regime/` | `local_supervision_regime` | client local step이 labeled/unlabeled/server-labeled regime 중 무엇을 쓰는지 |
| `server_step_policy/` | `server_step_policy` | server-side supervised seed step 같은 추가 server step 여부 |
| `peer_context_policy/` | `peer_context_policy` | client 간/round 간 helper context 교환 mechanism |
| `update_partition_policy/` | `update_partition_policy` | unified update인지 method-owned partitioned update인지 |
| `aggregation_weight_policy/` | `aggregation_weight_policy` | FedAvg류 aggregation weight 기준 |
| `query_multiview_source/` | `query_multiview_source` | weak/strong view가 materialized row에서 오는지, live agent가 만들지 |

`shard_policy`는 unlabeled/client pool의 non-IID 분배 방식을 소유하고,
`labeled_exposure_policy`는 선택된 labeled seed가 어느 boundary에 노출되는지를
소유한다. 즉 `shared_client_seed`는 shard policy가 아니라 exposure policy다.
현재 entrypoint 기본 exposure는 `shared_client_seed`다. `client_local_split`은
legacy/ablation으로 남긴다. `server_only_seed`는 materialized artifact와 run request
metadata까지는 열려 있지만, 실제 simulation 실행은 method-owned descriptor,
`server_step_policy=supervised_seed_step`, client-unlabeled regime, server step runtime이
붙기 전까지 compatibility validator가 막는다.

나머지 capability 축은 FedMatch 전용이 아니라 FL SSL 공통 조합 표면이다. 예를 들어
client participation은 `all_clients`, `fraction_random`, `fixed_count_random` 중에서
고르고, aggregation weight는 `example_count`, `uniform`, `accepted_count` 중에서 고른다.
FedMatch는 이 공통 축 중 `partitioned` update capability와 `uniform` aggregation
weight를 요구하는 method descriptor로 표현된다. `sigma/psi` partition 이름과 loss
routing 의미는 FedMatch method package가 소유한다.
마찬가지로 `prediction_similarity_topk`는 공통 peer-context mechanism만 표현하고,
FedMatch의 `num_helpers=2`, `h_interval=10` 같은 값은 FedMatch descriptor와
method package가 소유한다.

## `fl_method` 실행 계획

FL entrypoint는 `fl_method` section을 `FederatedSslExecutionPlan`으로 해석한다.
이 plan은 사람이 고른 상위 method와 runtime capability를 bootstrap 전에 검증하고,
report protocol에 남긴다.

- `composition_mode=method_owned`: FedMatch/FedLGMatch 같은 상위 method가 client
  objective, server policy, round state exchange 요구사항을 소유한다. 이 모드에서는
  lower axes를 따로 쓰지 않는다.
- `composition_mode=manual`: 논문 method가 아니라 `client_ssl_objective`,
  `server_aggregation`, `update_family`를 직접 조합하는 baseline/ablation 모드다.
  사용자는 `query_ssl_method`, `local_update_profile`,
  `round_runtime.adapter_family_name`, `round_runtime.aggregation_backend_name`을
  직접 고른다. lower axes는 compose된 leaf에서 실행 계획 builder가 자동
  파생한다. report/index에는 `execution_role=manual_baseline`으로 남고
  `descriptor_name`은 비워 둔다.

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

`method_descriptor`는 논문 method의 identity, report role, custom runtime 필요
여부와 원본 parameter snapshot 사용 여부를 표현한다. 원본 상세값 자체는
`methods/federated_ssl/<method>/original_spec.py`가 소유하고, YAML은
`scenario`, `use_original_parameters`, `parameter_overrides` 같은 실행 표면만 둔다.
실제 local update 계산 조합은 `local_update_profile`, server round runtime 조합은
`round_runtime.*` leaf에서 온다.

따라서 새 논문 method를 추가할 때는 descriptor config만 추가하지 않는다.
`docs/contracts/fl_ssl_method_capability_matrix.md`에서 capability 요구사항을 먼저
정리하고, 선택 전에는 `methods/federated_ssl/<method>/` 구현 폴더나
`method_descriptor/<method>.yaml` placeholder를 만들지 않는다. FedMatch는 첫 method로
선택되어 capability surface와 원본 core/config snapshot이 열린 예외다.
선택된 method의 descriptor/recipe metadata와 필요한 methods core를 구현한 뒤 이 config group을 연다.
method-only local/server/aggregation 변형은 method 폴더에 둘 수 있고,
`agent`/`main_server`에는 capability adapter만 둔다.

`tests/architecture/test_layer_dependencies.py`는 `method_descriptor/*.yaml`과 실제
`methods/federated_ssl/<method>/` 필수 파일이 일치하는지 검증한다.

사람이 읽는 method 조립표는 `methods/federated_ssl/<method>/descriptor.py`의 recipe
metadata가 소유하고, 커질 때만 optional `recipe.py`로 분리한다. Hydra YAML은 실행
조합과 override hook만 소유하며, 논문 원본 기본값을 복제하지 않는다.

## Compose Preset 제거 기준

method/local update/server round를 한꺼번에 고르는 high-level compose preset은 중복
source-of-truth를 만들기 때문에 두지 않는다. 논문 method 실행은 기본
`fl_method.composition_mode=method_owned`에서 `method_descriptor`와 method-owned
recipe 검증으로 닫고, lower-axis baseline/ablation은 `composition_mode=manual`에서
`local_update_profile`과 `round_runtime.*` leaf를 직접 override한다.
