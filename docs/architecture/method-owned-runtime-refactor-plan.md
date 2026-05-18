# Method-Owned Runtime Refactor Plan

이 문서는 FL SSL method framework를 `methods/` 중심으로 깊게 만들기 위한 active
리팩터링 계획이다. 목적은 코드 길이 절감이 아니라 새 method 추가 시 수정 위치가
예측 가능해지고, `agent`/`main_server` runtime이 method별 구현을 흡수하지 않게
하는 것이다.

## 목표 경계

```text
shared
  contracts, domain entity, canonical payload 해석

methods
  method identity, algorithm core, SSL hook, local objective,
  adaptation update core, aggregation math, server/round policy 의미

conf
  Hydra 실행 조합과 실행값 source of truth

agent
  private/local state, raw text 접근, local artifact materialization,
  selected method core 실행 port, payload upload

main_server
  round lifecycle, validation, repository/materialization,
  selected method/server policy 실행 port, publication

scripts
  experiment entrypoint, sweep, report/artifact orchestration thin wrapper
```

논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다. 이
폴더가 descriptor, recipe metadata 또는 optional recipe, local objective,
server/round policy, method-only aggregation 변형을 묶는다. 두 개 이상 방법론에서
공유되는 평균/투영/adapter payload 해석은 `methods/federated/aggregation/*` 또는
`methods/adaptation/<family>/*`로 승격한다.

새 method를 추가하기 위해 `agent`나 `main_server`에 method 이름을 가진 파일이
추가되면 실패 신호로 본다. runtime 계층에 추가할 수 있는 것은 method가 아니라
`raw_text_local_update`, `artifact_ref_materializer`, `round_state_exchange`,
`server_policy_executor` 같은 capability다.

## 수정된 Batch 순서

### Batch 0. WIP 정리와 registry primitive 잠금

상태: 완료. methods/main_server/agent 주요 registry 일부를 primitive와
implementation-local registration으로 정리했다.

남은 원칙:

- `registry.py`는 lookup/catalog/duplicate guard만 맡긴다.
- concrete builtin 목록을 `registry.py`나 `builtin_loader.py`에 계속 누적하지 않는다.
  name-to-module convention, config-declared module path, package manifest 중 하나로
  import trigger를 작게 유지한다.
- shared canonical payload family만 explicit contract source-of-truth 예외다.

### Batch 1. Local Update Runtime Seam 교정

상태: 1차 완료. Registry primitive, implementation-local catalog/registration,
selection hook 중복 제거, method-specific runtime guard를 닫았다. 단,
`builtin_loader.py` 제거는 Batch 1.5에서 별도 진행한다.

- acceptance policy는 SSL selection hook과 중복 판단을 하지 않고 metadata/compatibility
  역할로 축소한다.
- evidence/input/scoring registry는 implementation-local registration으로 정리한다.
- local training registry metadata는 implementation 옆 registry metadata를 source로
  두고, 별도 UI catalog snapshot을 source of truth로 삼지 않는다.
- `agent` local update backend는 method identity를 소유하지 않는 runtime adapter로
  명명/문서화한다.
- architecture guard로 `agent`/`main_server` method-specific runtime module 추가를 막는다.

완료 기준:

- 새 FL SSL method 추가를 위해 `agent`에 method-specific backend 파일을 추가하지 않는
  규칙이 문서와 테스트에 남는다.
- 기존 local training/scoring/selection 테스트가 통과한다.

### Batch 1.5. Builtin Loader 제거

상태: 완료. `agent` local runtime registries는 `builtin_loader.py`를 제거하고
package-local convention import helper와 `RuntimeRegistry` primitive로 전환했다.

`builtin_loader.py`는 registry 하단 등록 block을 제거하기 위한 중간 발판이다. 최종
구조에서는 concrete module 목록을 loader에 계속 누적하지 않는다.

- 완료: `agent` acceptance/evidence/input/scoring/training registry의 import trigger를
  convention 기반 package-local import로 바꿨다.
- 완료: `agent` registry 파일의 반복 dict/register/list/catalog 코드를
  `agent/src/services/runtime_registry.py`로 공통화했다.
- 완료: `main_server` aggregation registry의 import trigger를 aggregation method
  module convention으로 바꿨다. adapter family별 aggregation module은 두지 않는다.
- 완료: `methods/federated/aggregation` import trigger를
  adapter_kind/method_name convention과 bounded package import로 바꿨다.
- 완료: `methods/ssl/hooks` import trigger를 bounded package import로 바꿨다.
- shared canonical contract family의 explicit loader만 예외로 남긴다.

완료 기준:

- 일반 runtime/strategy registry에는 concrete 목록을 가진 `builtin_loader.py`가 남지
  않는다.
- 새 implementation 추가 시 registry나 loader 목록을 수정하지 않는다.

### Batch 2. Agent Local Update Executor Port

상태: 완료. `LocalTrainingService`에서 update 생성, privacy protection,
payload 저장, submission envelope 생성을 분리해 `LocalUpdateExecutor` port로 내렸다.
서비스는 model revision 검증, runtime compatibility 검증, pseudo-label selection,
accepted example 조립까지만 orchestration한다.

Batch 1에서 정리한 registry를 바탕으로 agent runtime seam을 더 깊게 만든다.

- 완료: `LocalTrainingService`가 concrete training backend default를 직접 import하지
  않고 `LocalUpdateExecutor`를 통해 selected backend/privacy guard를 resolve한다.
- 완료: architecture guard로 `LocalTrainingService`가 concrete training backend나
  training backend registry를 직접 import하지 못하게 했다.
- 완료: pseudo-label acceptance policy metadata를 agent-local registry에서 제거하고
  `methods/ssl/hooks/acceptance.py`로 이동했다. `agent`는 methods-owned
  selection/acceptance spec을 local candidate/context 조립에만 연결한다.
- 완료: family-specific scoring/privacy guard source를 agent에서 제거했다.
  classifier-head logits scoring은 `methods/adaptation/classifier_head/scoring.py`,
  clip-only privacy guard는 `methods/adaptation/privacy_guards/`가 소유하고,
  agent는 generic runtime bridge와 실행 흐름만 맡는다.
- 완료: LoRA-classifier training row, label schema resolve, train executor contract,
  delta payload 통계 조립을 `methods/adaptation/lora_classifier/update/local_update.py`로
  내렸다. `agent`에는 accepted example에서 raw text를 꺼내는 adapter와 agent-local
  artifact ref glue만 남겼다.
- 완료: privacy guard는 `methods/adaptation/privacy_guards/`가 소유한다.
  `agent`에는 privacy guard package를 두지 않고, `LocalUpdateExecutor`가
  selected guard를 methods-owned registry에서 resolve한다.
- 완료: diagonal-scale heuristic과 LoRA-classifier concrete training backend를
  `methods/adaptation/<family>/training_backend.py`로 이동했다. agent
  `training/backends/training` compatibility facade도 제거했고, local update
  backend registry는 `methods/adaptation/local_update_registry.py`를 직접 사용한다.
- `methods/adaptation/<family>/`가 local update 계산 core와 method-local runtime
  descriptor를 소유한다. shared payload family 파일은 payload shape와
  adapter_kind/update payload format만 소유한다.
- `agent`는 raw text, private storage, artifact ref, payload upload를 methods core에
  연결하는 generic executor port를 둔다.
- diagonal-scale heuristic과 LoRA classifier는 각각 methods-owned backend를 통해
  실행되고, agent는 capability port만 유지한다.
- LoRA/backbone/tokenizer spec 중 공통 의미는 methods/shared contract 쪽 typed spec으로
  모으고, runtime-specific artifact prefix나 bootstrap ref는 agent/scripts에 남긴다.

완료 기준:

- 새 local objective나 adaptation family를 추가할 때 agent 변경이 capability 확장인지
  method-specific 누수인지 구분된다.
- 기존 payload/update manifest shape는 유지된다.

### Batch 3. Profile Source Of Truth와 Compatibility

상태: 완료. Batch 번호는 계획 식별자일 뿐이고, 실제 다음 단계는 아래
handoff 순서를 따른다.

- Hydra leaf config를 실행값 source of truth로 유지한다.
- Python profile/mapping은 resolver, validator, legacy compatibility facade, 테스트 fixture
  helper 중 하나로 역할을 좁힌다. 실행 기본값을 Python mapping에 다시 쓰지 않는다.
- LocalUpdateProfile, method descriptor, payload family contract,
  `round_runtime.adapter_family_name`, `round_runtime.aggregation_backend_name`을 함께
  검증한다.
- incompatible 조합은 실행 중간이 아니라 bootstrap/resolve 시점에 실패한다.
- 닫힘: `methods/federated_ssl/training_algorithm_profiles.py`는 삭제했다. FL local
  update profile 실행값은 `conf/strategy_axes/fl/local_update_profile/*.yaml`이
  소유하고, Python은 `LocalUpdateProfile` parser/validator만 둔다.
- 닫힘: `shared/src/contracts/training_contracts.py`에서 method-specific
  `diagonal_scale_heuristic` backend default를 제거했다. `training_backend_name`은
  contract 필수 값이고, 실험 기본값은 Hydra local update profile이 소유한다.
- 최신화: `methods/federated_ssl/<method>/descriptor.py`의 recipe metadata 경계는
  method-owned 논문 method 전용이다. 기본 manual baseline은 descriptor를 두지 않고
  `query_ssl_method/local_update_profile/round_runtime.*` lower-axis 조합으로 실행한다.
- 닫힘: compatibility validator가 method descriptor뿐 아니라 recipe metadata,
  local update profile, adapter family, aggregation backend를 bootstrap 전에 함께
  검증한다.
- 닫힘: `FederatedSslExecutionPlan`을 추가해 `fl_method.composition_mode`를
  method-owned/manual로 해석하고, `security_policy=plaintext`만 현재 지원 capability로
  bootstrap 전에 검증한다. manual mode는 lower-axis 조합 baseline/ablation임을
  report protocol에 남긴다.
- 닫힘: `methods/federated_ssl/runtime_fallbacks.py`를 명시 runtime/API fallback
  module로 열고, 과분리였던 `training_defaults.py`와 `training_default_values.py`는
  제거했다. runtime 내부 코드는 `RUNTIME_FALLBACK_*` 이름을 직접 import한다.

다음 단계 handoff:

1. Batch 4-A aggregation ownership correction은 완료됐다. main_server family별 FedAvg
   구현과 aggregation projection은 methods 계층으로 이동했다.
2. Batch 3 profile/default 작업은 완료됐다. Batch 4-B의 기본/no-op server
   policy executor와 client metric summary round-state exchange seam도 닫았다.
   복잡한 custom state machine은 실제 method descriptor가 들어올 때 capability
   확장으로 다룬다.

시작 전 조건:

- `shared/src/config`를 다시 source-of-truth로 되살리지 않는다.
- `agent`/`main_server`에 method-specific 파일을 추가하지 않는다.
- 새 paper method 구현은 아직 하지 않는다. 다음 profile 작업은 조합 검증 seam을 닫는
  단계다.

첫 작업 순서:

1. Hydra profile과 Python mapping/default를 inventory한다.
   대상은 `conf/strategy_axes/fl/**`, `methods/federated_ssl/local_update_profile.py`,
   `methods/federated_ssl/runtime_fallbacks.py`다.
2. 실행값 source-of-truth 표를 만든다. threshold, scorer, evidence backend,
   privacy guard, local update backend, adapter family, aggregation backend가 어디에서
   오는지 한 곳에 고정한다.
3. 완료: profile resolve output을 typed object로 정의한다. 이 object는 runtime 실행값을 새로
   만들지 않고 Hydra config, method recipe metadata, descriptor metadata를 검증한 결과만 담는다.
4. 완료: compatibility validator를 최소 구현한다. method descriptor, recipe metadata, payload
   family contract, local update profile, adapter family, aggregation backend,
   runtime capability를 함께 확인한다.
5. 완료: method-only 변형은 `methods/federated_ssl/<method>/descriptor.py`의 recipe
   metadata 또는 optional `recipe.py`에서 선언하고, 재사용 backend/projection은 축별
   methods 패키지에서 참조하게 한다.
6. 완료: Python default/mapping module을 `runtime_fallbacks.py` 하나로 좁혔다.
   `training_defaults.py`와 `training_default_values.py`는 제거했고,
   `agent`/`main_server`/`scripts`는 `runtime_fallbacks.py`를 import한다.

검증 대상:

- `tests/unit/test_methods_federated_ssl.py`
- `tests/unit/test_scripts_hydra_configs.py`
- profile compatibility 신규 unit test
- `tests/architecture/test_layer_dependencies.py`
- 필요 시 2 clients / 1 round FL SSL smoke 또는 Hydra compile smoke

완료 기준:

- FedMatch/FedLGMatch류 조합을 추가할 때 method 폴더 중심으로 읽히고, runtime core
  수정이 아니라 profile/descriptor/recipe metadata 조합 검증으로 실패/성공이
  결정된다.
- 같은 실행 기본값이 Hydra와 Python에 중복 정의되지 않는다.
- incompatible method/profile 조합은 agent/main_server runtime이 호출되기 전에 실패한다.
- 새 profile 추가 시 수정 위치가 `conf`, method descriptor/profile validator, 테스트로
  예측 가능하다.

### Batch 4. Main Server Round/Policy Seam

상태: 4-A 완료, 4-B 기본 seam 완료. round family concrete module은 generic runtime으로 접었고,
aggregation method의 generic core는 `methods/federated/aggregation/`, adapter-family
FedAvg core/materialization은 각 family-owned
`methods/adaptation/<family>/aggregation/fedavg.py`와 필요 시
`server_preflight.py`로 옮겼다.
`main_server` aggregation package에는 generic executor/registry와 server-owned artifact
ref 생성/JSON loading capability만 남긴다. `assets/prototypes`는 catch-all assets package에서
`federation/prototypes` server-owned artifact lifecycle package로 좁혔다.
method-owned local update/scoring/privacy guard 항목은 methods registry로 위임했고,
runtime adapter는 agent-local example/evidence capability만 노출한다.

`main_server`에도 agent와 같은 문제가 생길 수 있으므로 별도 batch로 다룬다.

- `main_server/src/services/federation/rounds/families/registry.py`를 shared adapter
  payload registry 기반 generic runtime으로 유지한다.
- aggregation method 파일은 `main_server`에 두지 않는다. `main_server`는
  `methods/federated/aggregation/`에서 선택된 strategy를 호출하는 executor만 둔다.
- adapter-family FedAvg core/materialization은
  `methods/adaptation/<family>/aggregation/fedavg.py`와 필요 시
  `server_preflight.py` 같은 family-owned module에 둔다.
- adapter family별 파일이나 family별 aggregation service/config class를 추가하지 않는다.
- generic aggregation method 산술/strategy wiring은 `methods/federated/aggregation/`에
  두고, adapter family별 delta 해석과 next-state 계산은
  `methods/adaptation/<family>/aggregation/`에 둔다.
- method-only aggregation 변형은 `methods/federated_ssl/<method>/aggregation.py` 또는
  `server_policy.py`에 둘 수 있지만, `main_server`에서는 등록된 strategy/capability로만
  호출한다.
- server-owned `aggregation_artifact::` JSON ref materialization은 main_server
  runtime capability다. LoRA/classifier delta 해석은
  `methods/adaptation/lora_classifier/aggregation/fedavg.py`와
  `server_preflight.py`가 맡고, `main_server` artifact store에는 family
  branch를 두지 않는다.
- server-owned prototype artifact lifecycle은 `federation/prototypes`가 소유하고,
  `shared`는 prototype payload contract/serialization만 소유한다.
- method-specific server/round policy 의미는
  `methods/federated_ssl/<method>/server_policy.py`,
  `round_policy.py`에 둔다.
- `main_server`에는 generic `server_policy_executor` 또는 `round_state_exchange`
  capability만 추가한다.

완료 기준:

- FedMatch/FedLGMatch server-side 차이가 `main_server` core 파일명이나 registry branch로
  새지 않는다.

1차 완료:

1. methods-owned `server_policy.py` / `round_policy.py` 의미를 main_server가
   method 이름 분기 없이 실행할 수 있는 최소 `server_policy_executor` capability로
   연결했다.
2. FedAvg pseudo-label은 no-op/default policy로 통과시키고, FedMatch/FedLGMatch류가
   요구하는 custom server/round policy는 아직 live runtime에서 거부한다.
3. `TRACEMIND_ROUND_METHOD_DESCRIPTOR`와 `ServerRoundRuntimeConfig`에 method
   descriptor 선택 경계를 열고, round lifecycle 내부에는 method-specific 문자열을
   넣지 않았다.

2차 진행 목표:

1. FedMatch/FedLGMatch류가 round별 pseudo-label statistics, client weighting,
   calibration state를 요구할 때 사용할 `round_state_exchange` capability를 추가한다.
2. custom server/round policy가 실제로 필요해지는 method descriptor가 들어오기 전까지는
   `DefaultServerPolicyExecutor`가 명시적으로 실패하게 둔다.

2차 완료:

1. `methods.federated_ssl.base.FederatedSslRoundStateExchangeSpec`에 method가 요구하는
   round state exchange 이름과 required client metric key를 선언하게 했다.
2. main_server에는 method 이름이 아니라 `round_state_exchange/executor.py` capability를
   추가했다. 기본 executor는 `none`과 `client_metric_summary`만 지원하고, custom
   exchange는 bootstrap/finalize에서 실패한다.
3. round state summary는 aggregation metric과 섞지 않고
   `RoundPublicationSummary.round_state_summary_metrics`에 분리해 남긴다.

### Batch 5. SSL Hook Bundle과 Training Template

상태: 완료. `SslObjectiveHooks` typed bundle과 FixMatch fixed-threshold hook 조합은
열려 있고, selection-set epoch history/summary record는
`methods/adaptation/common/training_history.py`로 공통화했다. fixed classifier와 LoRA
classifier가 공유하는 selection 평가, history, best checkpoint 복원 흐름은
`methods/adaptation/common/selection_training_loop.py`로 분리했다. prototype score
policy는 `policy_registry.py`와 `score_policies/`로 분리했고, 얇은 `policies.py`
facade는 제거했다. model input 처리와 runner orchestration은 각 family/script에 남긴다.

- 완료: USB식 장점인 hook 교체 구조를 TraceMind 방식으로 명시화한다.
- 완료: `SslObjectiveHooks` role-based bundle로 masking, pseudo-labeling,
  consistency loss를 교체한다.
- 완료: FixMatch는 fixed-threshold hook bundle로 표현하고, FreeMatch-like test-only
  masking hook으로 교체 가능성을 검증한다.
- 완료: fixed classifier와 LoRA classifier의 evaluation/checkpoint/history/training
  loop 중 selection-set 평가, history record, best checkpoint 복원처럼 반복되는
  template만 공통화했다.
- 완료: prototype score policy 구현과 registry를 분리하고, 새 policy는
  `methods/prototype/scoring/score_policies/<policy>.py`에서 구현 옆 decorator로
  등록한다.

완료 기준:

- 새 SSL algorithm 추가가 scripts runner 수정이 아니라 methods algorithm/hook 추가로
  보인다.
- training loop 공통화가 model input 차이를 숨기지 않고 adapter로 표현한다.

### Batch 6. Runtime Adapter와 Artifact Writer 정리

상태: 완료. artifact/report builder-writer 분리와 federated agent runtime adapter
package 분리는 완료됐고, selection diagnostics dataclass 직렬화 반복은
`agent/src/services/training/selection/diagnostics_serialization.py` helper로
정리했다.

- `scripts/runtime_adapters/*`는 request mapper, repository wiring, runtime bridge로
  나눈다.
- artifact/report builder와 writer/exporter를 분리한다.
- compatibility facade는 제거 기준과 public boundary 여부를 문서화한다.
- diagnostics dataclass의 manual `to_mapping()` 반복은 serializer helper로 줄였다.

완료 기준:

- runner는 orchestration 중심으로 남고 schema 의미나 algorithm 의미를 만들지 않는다.

### Batch 7. Extension Dry Run

상태: 완료. production registry에 장난감 method를 남기지 않고
`tests/fixtures/federated_ssl_dummy_method.py` test-only method로 descriptor,
recipe, round-state exchange, compatibility validator, registry 경계를 검증했다.

- production registry에 장난감 method를 남기지 않는 방식으로 dummy FL SSL method
  extension test를 유지한다.
- 새 method 추가 시 필요한 변경이 method-local module, descriptor/config/profile,
  필요한 capability adapter뿐인지 확인한다.
- shared contract golden fixture와 architecture guard를 함께 유지한다.

완료 기준:

- 두 번째/세 번째 method 추가 비용이 테스트로 관측된다.

## Main Server 현재 점검

현재 `main_server`는 일부 정리가 되었지만 같은 위험이 남아 있다.

- 좋은 상태: aggregation registry는 primitive와 implementation-local registration으로
  낮아졌다.
- 좋은 상태: generic aggregation method는 `methods/federated/aggregation/`, family별
  projection은 `methods/adaptation/<family>/`가 소유하고, main_server는 generic
  executor와 server-owned artifact ref capability만 둔다.
- 닫힘: `rounds/families/registry.py`는 concrete family 파일을 import하지 않고,
  shared adapter payload registry와 aggregation backend를 generic round family runtime으로
  조합한다.
- 닫힘: server-owned LoRA artifact-ref materializer/loader 1차 구현은 generic JSON
  artifact store와 aggregation context capability로 붙였다. `main_server`는
  `aggregation_artifact::` ref를 읽고, LoRA/classifier delta 의미는 methods
  projection이 해석한다.
- 닫힘: method-owned training/scoring/privacy guard 항목은 methods registry에서 읽는다.
- 닫힘: live server runtime의 method descriptor 기반 `server_policy_executor`와
  `round_state_exchange` capability는 기본/no-op 경로와 client metric summary 경로까지
  연결했다. 복잡한 FL method가 custom state machine을 요구하면 먼저 method-local
  descriptor/server policy를 추가하고, main_server에는 capability 이름의 executor만
  확장해야 한다.

따라서 `main_server`의 다음 리팩터링 목표는 registry 정리 자체가 아니라 method-specific
server 의미가 round lifecycle로 새지 않게 하는 것이다.

## 금지 규칙

- 새 method 때문에 `agent/src/**/<method>*.py`를 만들지 않는다.
- 새 method 때문에 `main_server/src/**/<method>*.py`를 만들지 않는다.
- `shared/src/config`를 새 source of truth로 되살리지 않는다. shared에는 contract,
  domain entity, canonical payload 해석만 둔다.
- runtime adapter가 threshold, server weighting, pseudo-label objective 같은 method
  policy를 판단하지 않는다.
- scripts runner가 algorithm-specific registry를 새로 소유하지 않는다.

예외는 새 runtime capability가 생긴 경우뿐이다. 예외도 method 이름이 아니라 capability
이름으로 표현하고, method 의미는 `methods/`에 남긴다.
