# 전략 추가 플레이북

## 목적

이 문서는 TraceMind에서 새 전략을 추가할 때
`무엇을 어디에 구현하고, 어떻게 연결하고, 무엇을 검증할지`
를 순서대로 설명하는 실전용 플레이북이다.

[algorithm_extension_guide.md](algorithm_extension_guide.md)가
전략 축과 출발 파일을 알려주는 지도라면,
이 문서는 실제 작업 순서를 알려주는 매뉴얼이다.

현재 어떤 전략 축이 이미 활성 runtime인지,
어떤 축이 metadata only인지 보려면
[docs/strategy_surface_map.md](../strategy_surface_map.md)를
먼저 본다.

## 1. 먼저 변경 종류를 분류한다

전략 추가 작업은 먼저 아래 셋 중 어디에 해당하는지 나눠야 한다.

| 변경 종류 | 대표 예시 | 보통 건드리는 경계 |
|---|---|---|
| 같은 계약 안의 새 구현체 추가 | 새 training backend, 새 scoring policy, 새 aggregation backend | `methods` core + capability runtime adapter |
| 기본 선택값만 변경 | 기본 scorer 변경, 기본 aggregation backend 변경 | `conf/`, `methods/federated_ssl/training_defaults.py`, 또는 `main_server/src/services/federation/rounds/runtime/config.py` |
| 새 adapter family 추가 | `diagonal_scale` 외 LoRA family 추가 | `shared` + `methods/adaptation` + capability runtime adapter |
| 새 논문 method 추가 | FedMatch, FedLGMatch, (FL)^2 | `methods/federated_ssl/<method>/` + `conf` |

판별 기준은 단순하다.

1. payload shape가 그대로면 대개 "같은 계약 안의 새 구현체 추가"다.
2. 구현체는 그대로고 기본 fallback만 바꾸면 "기본 선택값만 변경"이다.
3. state/update payload가 달라지면 "새 adapter family 추가"다.
4. 논문 단위로 local objective, server/round policy, method-only aggregation 변형이
   묶이면 `methods/federated_ssl/<method>/`를 먼저 만든다.

## 2. 공통 작업 순서

거의 모든 전략 추가는 아래 순서를 따른다.

1. 바뀌는 축을 확정한다.
   - `training_backend`, `example_generation_backend`, `scorer_backend`,
     `score_policy`, `privacy_guard`, `aggregation_backend`,
     `prototype_builder`, `adapter_family` 중 무엇인지 먼저 적는다.
2. contract 변경이 필요한지 판별한다.
   - 새 payload shape가 없으면 `shared/src/contracts/`는 건드리지 않는다.
   - 새 family면 `shared/src/contracts/adapter_contract_families/`부터 본다.
3. concrete 구현을 추가한다.
   - 구현은 해당 소유 경계 파일에 넣는다.
4. thin registry wiring에 등록한다.
   - 새 코드에서는 구현 module 옆 decorator와 convention/config 기반 import trigger를 우선한다.
   - `registry.py` 하단에 concrete 등록 block을 계속 누적하지 않는다.
5. compatibility와 default를 분리해서 다룬다.
   - 새 구현을 추가하는 것과 기본 선택값을 바꾸는 것은 다른 작업이다.
6. unit test를 같은 경계에서 닫는다.
   - producer/consumer가 같은 의미를 보는지 같이 검증한다.
7. 문서를 최소 한 곳 갱신한다.
   - `algorithm_extension_guide.md`나 이 플레이북에 새 이름/축을 반영한다.

## 3. 같은 계약 안의 새 구현체 추가

### 3-0. 새 prototype builder 추가

예: `dbscan`, `kmeans` 변형

보통 수정 파일:

- [methods/prototype/building/base.py](../../methods/prototype/building/base.py)
- [methods/prototype/building/single.py](../../methods/prototype/building/single.py)
- [methods/prototype/building/kmeans.py](../../methods/prototype/building/kmeans.py)
- [methods/prototype/building/dbscan.py](../../methods/prototype/building/dbscan.py)
- [methods/prototype/building/pack_builder.py](../../methods/prototype/building/pack_builder.py)
- [scripts/experiments/prototype_analysis/prototype_strategy/strategies.py](../../scripts/experiments/prototype_analysis/prototype_strategy/strategies.py)
- 운영 artifact builder로 열 때만 [conf/strategy_axes/prototype/build_strategy/](../../conf/strategy_axes/prototype/build_strategy/)

작업 순서:

1. Prototype pack payload shape가 기존 `PrototypePackPayload`로 충분한지 먼저 확인한다.
2. build 알고리즘 계산은 `methods/prototype/building/<builder_name>.py`에 둔다.
3. 실험 모듈에는 `PrototypeIndex` adapter와 selection glue만 둔다.
4. exact incremental build-state를 지원하지 않으면 `supports_exact_build_state=False`로 둔다.
5. 기본 artifact builder로 노출할 때만 `conf/strategy_axes/prototype/build_strategy/`에 Hydra group을 추가한다.

우선 볼 테스트:

- [tests/unit/test_prototype_build_strategies.py](../../tests/unit/test_prototype_build_strategies.py)
- [tests/unit/test_prototype_strategy_experiment.py](../../tests/unit/test_prototype_strategy_experiment.py)

### 3-1. 새 training backend 추가

예: `diagonal_scale_gradient`

핵심 질문:

- update payload shape가 기존 `VectorAdapterDelta`와 같은가
- adapter family는 그대로 `diagonal_scale`인가
- backend별 세부 파라미터가 필요한가

보통 수정 파일:

- [methods/adaptation/](../../methods/adaptation/)
- [methods/adaptation/local_update_backend.py](../../methods/adaptation/local_update_backend.py)
- [methods/adaptation/local_update_registry.py](../../methods/adaptation/local_update_registry.py)
- 필요 시 [agent/src/services/training/execution/runtime_compatibility.py](../../agent/src/services/training/execution/runtime_compatibility.py)
- 기본값을 바꿀 때만 [methods/federated_ssl/training_defaults.py](../../methods/federated_ssl/training_defaults.py)

작업 순서:

1. 재사용 가능한 update 계산은 `methods/adaptation/<adapter_family>/`에 추가한다.
2. concrete backend는 `methods/adaptation/<family>/training_backend.py`에 둔다.
3. agent에는 method-specific 파일을 추가하지 말고 local runtime capability adapter만 둔다.
4. `SharedAdapterTrainingBackend` Protocol을 만족하는 adapter가 필요하면 capability 이름을 쓴다.
5. backend가 objective별 설정을 읽어야 하면 `from_objective_config(...)`를 둔다.
6. backend instance 재사용이 설정에 따라 달라지면 `matches_objective_config(...)`를 구현한다.
7. 구현 module 옆 decorator와 convention/config 기반 import trigger로 registry에 연결한다.
8. 새 backend가 기존 example/scorer/privacy 조합과 다르면 compatibility를 확인한다.
9. 기본값까지 바꾸려면 Hydra profile source of truth와 compatibility facade 범위를 확인한다.

보통 건드리지 않는 것:

- `main_server` aggregation
- `shared/src/contracts/adapter_contract_families/`

우선 볼 테스트:

- [tests/unit/test_methods_diagonal_scale_heuristic_update.py](../../tests/unit/test_methods_diagonal_scale_heuristic_update.py)
- [agent/tests/unit/test_local_training_service.py](../../agent/tests/unit/test_local_training_service.py)
- 필요 시 [agent/tests/unit/test_training_example_service.py](../../agent/tests/unit/test_training_example_service.py)

### 3-2. 새 example-generation backend 추가

예: cached score 재사용 backend, feedback-only backend

보통 수정 파일:

- [methods/prototype/training_inputs/](../../methods/prototype/training_inputs/)
- [agent/src/services/training/backends/inputs/registry.py](../../agent/src/services/training/backends/inputs/registry.py)
- [agent/src/services/training/examples/service.py](../../agent/src/services/training/examples/service.py)
- 필요 시 [agent/src/services/training/execution/runtime_compatibility.py](../../agent/src/services/training/execution/runtime_compatibility.py)

작업 순서:

1. 재사용 가능한 prototype input view 계산은 `methods/prototype/training_inputs/`에 추가한다.
2. `TrainingExampleBackend` 구현 클래스를 추가한다.
3. `supported_adapter_kinds`를 정확히 적는다.
4. 구현 module 옆 decorator와 convention/config 기반 import trigger로 등록한다.
5. `resolve_training_example_backend(...)` 경로에서 현재 training backend와 호환되는지 확인한다.
6. stored event 경로와 raw row 경로를 둘 다 테스트한다.

우선 볼 테스트:

- [tests/unit/test_methods_prototype_training_inputs.py](../../tests/unit/test_methods_prototype_training_inputs.py)
- [agent/tests/unit/test_training_example_service.py](../../agent/tests/unit/test_training_example_service.py)
- [agent/tests/unit/test_local_training_service.py](../../agent/tests/unit/test_local_training_service.py)

### 3-3. 새 scorer backend 또는 scoring policy 추가

보통 수정 파일:

- [methods/prototype/scoring/](../../methods/prototype/scoring/)
- [agent/src/services/inference/scoring_backends/](../../agent/src/services/inference/scoring_backends/)
- 필요 시 [scripts/experiments/prototype_analysis/prototype_strategy/scoring.py](../../scripts/experiments/prototype_analysis/prototype_strategy/scoring.py)

작업 순서:

1. scorer backend를 추가할지, 기존 backend 안의 policy만 추가할지 먼저 나눈다.
2. prototype score 계산/policy는 `methods/prototype/scoring/`에 추가한다.
3. backend면 구현 module 옆 decorator와 convention/config 기반 import trigger로 등록하고, policy면
   `methods/prototype/scoring/` registry 규칙을 따른다.
4. backend면 `confidence_kind`를 같이 선언해 pipeline/query buffer가 이름 분기 없이 읽게 한다.
5. `supported_adapter_kinds`가 달라지면 runtime validation에서 조합이 통과하는지 확인한다.
6. 실험에서도 같은 축을 쓸 거면 `prototype_strategy/scoring.py`도 같이 맞춘다.

우선 볼 테스트:

- [tests/unit/test_methods_prototype_scoring.py](../../tests/unit/test_methods_prototype_scoring.py)
- [agent/tests/unit/test_scoring_service.py](../../agent/tests/unit/test_scoring_service.py)
- [tests/unit/test_prototype_strategy_experiment.py](../../tests/unit/test_prototype_strategy_experiment.py)

### 3-4. 새 aggregation backend 추가

예: 같은 `diagonal_scale` family 안에서 robust aggregation backend 추가

보통 수정 파일:

- [methods/federated/aggregation/](../../methods/federated/aggregation/)
- `methods/adaptation/<family>/`
- 기본 backend를 바꿀 때만 [main_server/src/services/federation/rounds/runtime/config.py](../../main_server/src/services/federation/rounds/runtime/config.py)

작업 순서:

1. 먼저 새 로직이 method-only 변형인지 재사용 backend인지 판단한다.
2. FedMatch 같은 논문 방법론에만 종속된 client weighting, state exchange,
   pseudo-label 통계 결합은 `methods/federated_ssl/<method>/aggregation.py`,
   `server_policy.py`, `round_policy.py`에 둔다.
3. 두 개 이상 방법론에서 공유되는 backend의 generic 산술/strategy wiring은
   `methods/federated/aggregation/`에 추가한다.
4. adapter family별 delta 해석과 next-state projection은 `methods/adaptation/<family>/`에 둔다.
5. family projection에서 methods-owned aggregation strategy를 등록한다.
6. 현재 family runtime이 새 backend 이름을 selected strategy로 resolve하는지 확인한다.
7. 기본 선택값까지 바꾸려면 `ServerRoundRuntimeConfig.aggregation_backend_name` fallback을 수정한다.

보통 건드리지 않는 것:

- `agent` local runtime
- `shared/src/contracts/adapter_contract_families/`

우선 볼 테스트:

- [main_server/tests/unit/test_aggregation_service.py](../../main_server/tests/unit/test_aggregation_service.py)
- [main_server/tests/unit/test_round_runtime_factory.py](../../main_server/tests/unit/test_round_runtime_factory.py)
- [tests/unit/test_methods_fedavg.py](../../tests/unit/test_methods_fedavg.py)

## 4. 새 adapter family 추가

이 경우만 blast radius가 크다.
이 작업은 구현체 하나 추가가 아니라 새 계약과 새 조립 경로를 여는 일이다.

보통 수정 파일:

- [shared/src/contracts/adapter_contract_families/](../../shared/src/contracts/adapter_contract_families/)
- [methods/adaptation/<family>/training_backend.py](../../methods/adaptation/)
- [agent/src/services/training/execution/privacy_guards/](../../agent/src/services/training/execution/privacy_guards/)
- [agent/src/services/training/execution/runtime_compatibility.py](../../agent/src/services/training/execution/runtime_compatibility.py)
- [main_server/src/services/federation/rounds/aggregation/registry.py](../../main_server/src/services/federation/rounds/aggregation/registry.py)
- [main_server/src/services/federation/rounds/boundary/mappers.py](../../main_server/src/services/federation/rounds/boundary/mappers.py)

작업 순서:

1. 새 state/update payload를 `shared`에 정의한다.
2. 해당 family payload 파일 옆에 canonical `adapter_kind`와 update payload format
   상수를 둔다. 중앙 adapter family metadata catalog는 추가하지 않는다.
3. agent training backend와 privacy guard가 새 `adapter_kind`를 생산/소비하게 한다.
   단, method-specific 의미는 agent에 두지 않고 capability adapter로만 연결한다.
4. example backend, scorer, acceptance policy가 새 family를 지원하는지 확인한다.
5. 필요하면 `methods/adaptation/<family>/aggregation_projection.py` 또는 method-local
   recipe/aggregation이 해당 family projection을 지원하게 한다.
6. shared payload registry와 aggregation backend만으로 generic round family runtime이
   조립되는지 확인한다. `main_server/.../families/`에 family-specific 파일을 추가하지
   않는다.
7. payload 변환과 publication 경로를 점검한다.

이 경우는 producer와 consumer가 함께 바뀌므로
문서보다 코드 계약을 먼저 읽는 편이 맞다.

## 5. 기본값 변경과 새 구현 추가를 분리한다

새 구현을 추가했다고 기본이 바뀌어야 하는 것은 아니다.

### 기본값만 바꾸는 경우

- local training 기본 전략:
  - [methods/federated_ssl/training_defaults.py](../../methods/federated_ssl/training_defaults.py)
- server round runtime 기본 aggregation:
  - [main_server/src/services/federation/rounds/runtime/config.py](../../main_server/src/services/federation/rounds/runtime/config.py)
- experiment preset:
  - `conf/entrypoints/` 아래 Hydra config group

원칙:

- local training 기본 profile/default는 `methods/federated_ssl`
- 실제 실험 실행값의 source of truth는 `conf/`
- server-owned round runtime 기본값은 `main_server`
- 실험 preset 기본값은 `conf`

## 6. 전략 추가 후 검증 순서

최소 검증은 아래 순서로 한다.

1. registry/build 함수가 새 이름을 읽는지 확인
2. `from_objective_config(...)` 또는 runtime config loader가 새 이름을 조립하는지 확인
3. compatibility가 허용/거부를 올바르게 하는지 확인
4. 기존 기본 구현이 깨지지 않는지 회귀 확인
5. 실험 축을 같이 쓴다면 scripts 진입점도 확인

권장 대상 테스트:

- [agent/tests/unit/test_local_training_service.py](../../agent/tests/unit/test_local_training_service.py)
- [agent/tests/unit/test_training_example_service.py](../../agent/tests/unit/test_training_example_service.py)
- [agent/tests/unit/test_scoring_service.py](../../agent/tests/unit/test_scoring_service.py)
- [main_server/tests/unit/test_aggregation_service.py](../../main_server/tests/unit/test_aggregation_service.py)
- [main_server/tests/unit/test_round_runtime_factory.py](../../main_server/tests/unit/test_round_runtime_factory.py)
- [tests/unit/test_run_federated_simulation.py](../../tests/unit/test_run_federated_simulation.py)
- [tests/unit/test_prototype_strategy_experiment.py](../../tests/unit/test_prototype_strategy_experiment.py)

## 7. 빠른 판단 규칙

작업 중 아래 질문에 `예`가 나오면 범위를 넓혀야 한다.

1. 새 payload shape가 필요한가
2. agent와 server가 서로 다른 이름이나 필드 의미를 보게 되는가
3. 같은 설정 이름인데 family마다 해석이 달라지는가
4. 새 구현을 넣으려는데 기존 service를 concrete class `if`로 수정해야 하는가

이 중 하나라도 해당하면 단순 registry 추가가 아니라
계약 또는 compatibility 설계를 다시 봐야 한다.
