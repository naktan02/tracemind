# Pattern Integrity Refactor Backlog

이 문서는 패턴 이름은 붙어 있지만 책임 경계가 아직 충분히 깊지 않은 후보를 추적한다.
source of truth는 코드와 code-adjacent README이며, 이 문서는 다음 리팩터링 순서를 정하는
작업 목록이다.

## 현재 목표 경계

- `methods`는 production runtime과 scripts가 함께 쓰는 계산 core의 소유 계층이다.
- `conf`는 실행 조합과 파라미터만 소유한다.
- `agent`와 `main_server`는 runtime adapter/orchestrator로 선택된 core를 호출한다.
- `shared`는 contract, domain entity, canonical payload 해석만 소유한다.
- 논문 방법론은 `methods/federated_ssl/<method>/`를 사람이 읽는 시작점으로 둔다.
  method-only 변형은 이 폴더에 남길 수 있고, 두 개 이상 방법론에서 공유되는 계산은
  `methods/federated/aggregation/*`, `methods/adaptation/<family>/*`,
  `methods/ssl/*` 같은 축별 패키지로 승격한다.

## 2026-05-08 닫은 항목

- `methods/federated/aggregation/registry.py`: registry primitive에서 FedAvg concrete import와
  하단 등록 block을 제거했다. method metadata는 family-owned projection registration이
  제공하고, registry는 method name과 adapter kind convention으로 필요한 module만 import한다.
- `main_server/src/services/federation/rounds/aggregation/registry.py`: runtime aggregation
  backend registry를 primitive로 낮췄다. backend factory 등록은 각 backend module 옆 decorator가
  소유한다.
- `agent/src/services/training/backends/training/registry.py`: training backend registry에서
  concrete backend import와 하단 등록 block을 제거했다. 이후 concrete backend 구현도
  `methods/adaptation/<family>/training_backend.py`로 이동했고, agent registry는
  methods-owned registry facade만 맡는다.
- `methods/ssl/hooks/registry.py`: selection hook registry를 lookup/decorator primitive로
  낮췄다. 남은 import trigger cleanup은 별도 backlog에서 추적한다.
- `main_server/src/services/federation/rounds/families/registry.py`: registry 하단
  concrete family builder와 등록 block을 제거했다. 추가로 concrete 목록을 가진
  `builtin_loader.py` 대신 family name -> module name convention 기반 lazy import로
  바꿨다.
- `main_server/src/services/federation/rounds/families/`: concrete adapter family 파일을
  제거하고 shared payload registry + aggregation backend로 조합되는 generic
  `SharedAdapterRoundFamilyRuntime`만 남겼다.
- `main_server/src/services/federation/rounds/aggregation/registry.py`: concrete 목록을
  가진 `builtin_loader.py`를 제거하고 adapter_kind -> module name convention 기반
  lazy import로 바꿨다.
- `methods/federated/aggregation/registry.py`: concrete 목록을 가진 `builtin_loader.py`를
  제거하고 adapter_kind/method_name convention 기반 lazy import와 bounded package import로
  바꿨다.
- `methods/ssl/hooks/registry.py`: concrete 목록을 가진 `builtin_loader.py`를 제거하고
  bounded package import로 hook decorator registration을 실행한다.
- `agent` local runtime registries: training/evidence/input/acceptance/scoring registry에서
  concrete 목록을 가진 `builtin_loader.py`를 제거했다. package-local naming convention
  import helper로 필요한 module만 import하고, catalog listing은 package-local module을
  bounded import한다.
- `agent/src/services/runtime_registry.py`: agent local runtime registry의 반복 primitive
  코드(dict/register/get/list/catalog)를 공통화했다. 각 domain registry 파일은 public
  함수 이름과 factory 호출 방식만 남긴다.
- `agent/src/services/training/execution/local_training_service.py`: update 생성,
  privacy protection, payload 저장, submission envelope 생성을
  `LocalUpdateExecutor` port로 분리했다. `LocalTrainingService`는 selection
  orchestration만 맡고 concrete training backend나 training backend registry를 직접
  import하지 않는다.
- `methods/adaptation/lora_classifier/local_update.py`: LoRA-classifier local update의
  training row, label schema resolve, train executor contract, shared delta payload
  조립을 method-local core로 이동했다. `agent` lora backend는 raw text 추출과
  agent-local artifact ref glue를 methods core에 연결한다.
- `agent/src/services/training/execution/privacy_guards/`: privacy guard Protocol,
  concrete guard, registry, catalog entry를 한 파일에서 분리했다. registry는
  `RuntimeRegistry` primitive를 쓰고 guard 구현 옆 decorator가 factory/catalog를
  등록한다.
- `methods/federated/aggregation/fedavg/strategy.py`: FedAvg method 선택과 공통
  lineage/type 검증을 담당하는 generic strategy wiring으로 제한했다.
- `methods/adaptation/<family>/fedavg.py`, `fedavg_projection.py`: adapter family별
  FedAvg core, payload projection, 다음 state materialization을 family-owned module로
  이동했다.
- `main_server/src/services/federation/rounds/aggregation/artifact_refs.py`: LoRA-classifier
  aggregate artifact ref 생성 규칙을 server-owned capability helper로 분리했다.
  methods strategy는 logical artifact slot만 알고 실제 ref 생성은 main_server context가
  제공한다.
- `main_server/src/services/federation/rounds/aggregation/`: adapter family별
  `diagonal_scale.py`, `classifier_head.py`, `lora_classifier.py` module과 aggregation
  method file을 제거했다. `main_server`에는 `executor.py`, `registry.py`,
  `artifact_refs.py`만 남기고 새 aggregation method 추가가 main_server 파일 증가로
  이어지지 않게 architecture guard를 추가했다.
- `main_server/src/services/federation/prototypes/`: catch-all
  `federation/assets/prototypes`에서 server-owned prototype artifact lifecycle package로
  이름을 좁혔다. shared는 prototype repository/service가 아니라 payload contract와
  canonical serialization만 소유한다.
- `scripts/experiments/fl_ssl/federated_simulation/io/artifacts.py`: writer/builder를
  단순 위임하던 compatibility facade를 제거했다. `flow/`는 `RunArtifactWriter`,
  `SelectionDiagnosticsWriter`, `SimulationReportBuilder`를 직접 호출한다.
- `scripts/runtime_adapters/federated_server_runtime.py`: server runtime adapter
  package를 단순 re-export하던 compatibility facade를 제거했다. 호출부는
  `federated_server/runtime.py`, `initial_state_factory.py`,
  `round_request_mapper.py`를 직접 import한다.
- `scripts/reporting/classification_report.py`: shared canonical classification report
  utility를 단순 re-export하던 wrapper를 제거했다. scripts/tests는
  `shared.src.domain.services.classification_report`를 직접 import한다.
- `scripts/io/labeled_query_rows.py`: shared canonical labeled query row contract를
  단순 re-export하던 wrapper를 제거했다. scripts/tests는
  `shared.src.contracts.labeled_query_row_contracts`를 직접 import한다.
- `main_server/src/services/experiment_workspace/compiler/policies.py`: compile
  policy contract, registry primitive, default wiring, concrete entrypoint policy가
  한 파일에 섞여 있던 구조를 `contracts.py`, `registry.py`,
  `default_registry.py`, `entrypoint_policies.py`로 분리했다.

## Batch 1 상태와 잔여 항목

- `agent/src/services/training/acceptance_policies/`
  - 완료: acceptance policy를 SSL selection hook과 중복 판단하지 않는 metadata/compatibility
    seam으로 낮춘다.
  - 현재 기준: `registry.py`는 primitive, `top1.py`는 implementation-local catalog와
    decorator registration, selection 판단은 `methods/ssl/hooks/selection.py`가 맡는다.
- `agent/src/services/training/backends/evidence/`
  - 완료: evidence backend 구현 옆 factory/catalog 등록, registry primitive,
    resolver 분리.
- `agent/src/services/training/backends/inputs/`
  - 완료: input backend 구현 옆 factory/catalog 등록, registry primitive,
    compatibility/resolver 분리.
- `agent/src/services/inference/scoring_backends/`
  - 완료: monolith module을 package로 나누고 concrete scoring adapter는 구현 옆 decorator로
    등록한다.
- `agent/src/services/training/backends/training/`
  - 완료: concrete local update backend를 제거하고 registry facade만 남긴다.

잔여는 registry 모양이 아니라 agent runtime adapter가 methods core를 얼마나 얇게
호출하는지다. evidence/input/scoring adapter가 새 method 추가 때 agent 파일 증가로
이어지는지는 Batch 5-6에서 다시 점검한다.

## 다음 단계 Handoff

현재 작업 흐름은 Batch 4-A aggregation ownership correction이 먼저 완료된 상태다.
다음 작업자는 Batch 번호보다 아래 순서를 우선한다.

1. Profile source-of-truth와 compatibility validator를 정리한다. 실행값은 Hydra에 두고,
   Python은 resolver/validator/legacy facade/test fixture helper로 좁힌다.
2. Batch 4-B: server-owned LoRA artifact-ref materializer/loader 1차는 완료됐다.
   남은 server policy executor와 round state exchange capability를 추가할 때
   aggregation/round lifecycle로 method 의미가 새지 않게 한다.
3. Batch 5: SSL hook bundle 교체성, classifier selection-set history helper,
   prototype score policy package 분리는 1차 완료됐다. 다음은 training template
   잔여 반복 제거다.
4. Batch 6: scripts runtime adapter, artifact writer/exporter, diagnostics serializer를
   정리한다.
5. Batch 7: dummy method dry-run과 extension guard로 새 method 추가 비용을 검증한다.

## 다음 Profile 단계 체크리스트

다음 profile 단계는 패턴 정리가 아니라 실행 조합의 source-of-truth를 줄이는 작업이다.
목표는 Hydra config가 실행값을 소유하고, Python은 조합 해석과 실패 시점 검증만 맡게
만드는 것이다.

- `conf/strategy_axes/fl/**`와 `methods/federated_ssl/*profile*.py`,
  `methods/federated_ssl/training_default*.py`의 중복 실행값을 표로 만든다.
- threshold, scorer, evidence backend, privacy guard, local update backend, round runtime
  profile, aggregation backend의 canonical owner를 하나씩 정한다.
- Python module이 실행 기본값을 만들고 있으면 validator, resolver, legacy facade,
  test fixture helper 중 하나로 역할을 낮춘다.
- 완료: compatibility validator는 method descriptor와 recipe metadata,
  LocalUpdateProfile, adapter family, aggregation backend, round runtime profile을
  한 번에 본다.
- 완료: Python default module은 `runtime_fallbacks.py` 하나로 좁혔다.
- 실패는 training 중간이 아니라 profile resolve/bootstrap에서 발생해야 한다.

다음 profile 단계에서 하지 않는 일:

- FedMatch/FedLGMatch/(FL)^2 실제 학습 구현을 시작하지 않는다. 단, method 폴더 중심
  descriptor/optional recipe extension rule은 문서와 validator에 반영한다.
- `shared/src/config`에 새 profile/catalog/default source-of-truth를 만들지 않는다.
- `agent` 또는 `main_server`에 method 이름이 들어간 runtime 파일을 추가하지 않는다.
- runtime adapter가 method threshold, weighting, pseudo-label objective 의미를 판단하게
  하지 않는다.

다음 profile 단계 완료 기준:

- 같은 실행 기본값이 Hydra와 Python mapping에 중복되지 않는다.
- incompatible method/profile 조합을 profile resolve 단계에서 잡는 테스트가 있다.
- 새 FL SSL profile 추가 시 수정 위치가 `conf`, method descriptor/profile validator,
  tests로 제한된다.
- 기존 FL SSL smoke/report/manifest shape가 유지된다.

## Registry Backlog

- Import trigger cleanup
  - 문제: `builtin_loader.py`는 registry 하단 등록 block보다 낫지만, concrete module 목록을
    중앙 파일에 옮긴 것만으로는 최종 구조가 아니다.
  - 방향: name-to-module convention, config-declared module path, package manifest 중
    해당 축에 맞는 방식을 선택한다. shared canonical contract family만 explicit list 예외다.
  - 남은 대상: 새로 발견되는 일반 runtime/strategy registry. 현재 Batch 1.5 대상은 닫힘.
- `methods/prototype/scoring/policies.py`
  - 닫힘: `policy_registry.py`와 `score_policies/`로 분리하고, 얇은
    `policies.py` facade는 제거했다.
- `main_server/src/services/experiment_workspace/compiler/policies.py`
  - 닫힘: compile policy 구현, registry primitive, default wiring을 분리하고
    중앙 `policies.py` 파일을 제거했다.

## Non-Registry Pattern Backlog

- `agent/src/services/training/backends/training/`
  - 패턴: Runtime Adapter / Port
  - 진행: `LocalUpdateExecutor` port가 update submission 실행을 맡도록 분리했다.
  - 진행: LoRA-classifier label schema/train executor/payload stats core를 methods로
    이동했다.
  - 진행: diagonal-scale과 LoRA-classifier concrete training backend를
    `methods/adaptation/<family>/training_backend.py`로 이동했다.
  - 남은 방향: `agent`는 raw text/private state/artifact/payload upload capability만
    맡고, 새 backend 추가 시 `agent/src/services/training/backends/training/`에 파일을
    추가하지 않는다.
- `main_server/src/services/federation/rounds/`
  - 패턴: Runtime Adapter / Server Policy
  - 진행: adapter family별 round family 파일은 generic runtime으로 접었다.
  - 진행: FedAvg method 선택과 공통 lineage/type 검증은
    `methods/federated/aggregation/fedavg/strategy.py`로 이동했다.
  - 진행: adapter family별 FedAvg core와 payload projection은
    `methods/adaptation/<family>/fedavg.py`, `fedavg_projection.py`로 이동했다.
  - 진행: LoRA-classifier aggregate artifact ref 생성 규칙은 `artifact_refs.py` server
    capability로 분리했다. methods strategy는 logical artifact slot만 알고 실제 ref
    생성은 main_server context가 제공한다.
  - 진행: server-owned `aggregation_artifact::` JSON artifact-ref materializer/loader를
    generic aggregation context capability로 붙였다. LoRA/classifier delta 해석은
    methods projection이 맡는다.
  - 진행: main_server aggregation package는 generic executor/registry/artifact ref
    capability만 남겼다.
  - 진행: server-owned prototype artifact lifecycle은 `federation/prototypes`로 위치를
    명확히 했다.
  - 문제: FedMatch/FedLGMatch server-side policy와 round state exchange가 들어오면
    round lifecycle이나 aggregation adapter가 method 의미를 흡수할 수 있다.
  - 방향: method-specific server/round policy는 `methods/federated_ssl/<method>/`가 소유하고,
    method-only aggregation 변형도 method 폴더에 둘 수 있다. `main_server`에는 generic
    server policy executor와 round state exchange capability만 둔다.
- `main_server/src/services/experiment_workspace/catalog/local_training_registry_snapshot.py`
  - 패턴: Catalog/Facade
  - 진행: shared가 agent local training implementation module path를 들고 있던
    catalog snapshot을 main_server experiment workspace catalog로 이동했다.
  - 남은 문제: workspace UI/catalog 노출용 stable snapshot이므로 implementation
    registration source of truth가 되면 안 된다.
  - 방향: agent runtime registry wiring은 구현 옆 catalog entry를 사용한다. main_server
    snapshot은 read-only display facade로만 유지하고 새 method/experiment-only backend를
    여기에 누적하지 않는다.
- `methods/federated_ssl/runtime_fallbacks.py`
  - 패턴: Profile / Defaults
  - 진행: shared에서 제거해 FL SSL method/profile 계층인 `methods/federated_ssl`로
    이동했다. `training_algorithm_profiles.py`는 Hydra
    `conf/strategy_axes/fl/local_update_profile/*.yaml`과 중복되는 Python mapping
    catalog라 삭제했다. `shared` contract의 `diagonal_scale_heuristic` 기본 backend도
    제거해 `training_backend_name`을 명시 필수 값으로 되돌렸다.
  - 완료: production/API fallback은 `runtime_fallbacks.py`가 명시적으로 소유한다.
    과분리였던 `training_default_values.py`와 `training_defaults.py`는 제거했고,
    runtime 계층은 architecture guard로 legacy default module 재도입을 막는다.
  - 완료 기준: Python profile/default module이 새 실행 기본값 source-of-truth가 되지 않는다.
- `scripts/experiments/fl_ssl/federated_simulation/io/simulation_report_builder.py`
  - 패턴: Builder/Writer
  - 문제: report payload 조립과 파일 저장이 `save_report()` 안에 함께 있다.
  - 방향: builder는 payload dict만 만들고 writer가 path/JSON serialization을 맡긴다.
- `scripts/experiments/query_lora_ssl/io/teacher_pseudo_label_exporter.py`
  - 패턴: Builder/Exporter
  - 문제: teacher evidence/row 변환과 prediction artifact 쓰기가 같은 exporter에 있다.
  - 방향: `TeacherPseudoLabelBuilder`와 `TeacherPseudoLabelArtifactWriter`로 분리한다.
- `scripts/experiments/query_lora_ssl/io/artifacts.py`
  - 패턴: Writer/Exporter
  - 문제: run directory, adapter save, tokenizer save, classifier manifest, report write가 한 함수에
    모여 있다.
  - 방향: model artifact export, manifest build, report write를 분리한다.
- `scripts/runtime_adapters/federated_agent_runtime.py`
  - 패턴: Runtime Adapter
  - 문제: script runtime mapping, weak/strong row validation, training task 생성, backend resolve가
    한 adapter에 남아 있다.
  - 방향: request mapper, row validator, backend resolver를 package 내부 module로 분리한다.
- `main_server/src/services/experiment_workspace/compiler/service.py`
  - 패턴: Compiler / Patch Builder
  - 문제: workspace selection 검증, Hydra selector/override 조립, policy 실행,
    output payload 조립이 한 service method에 남아 있다.
  - 방향: policy monolith는 닫혔으므로 다음 정리는 selection compiler 또는
    Hydra override builder를 분리해 service가 orchestration만 맡게 한다.
- `scripts/experiments/prototype_analysis/prototype_strategy/sweep.py`
  - 패턴: Runner/Policy
  - 문제: threshold policy 실행, evaluation 선택, artifact write가 runner에 같이 있다.
  - 방향: experiment runner는 orchestration만 맡기고 evaluator, selection policy, writer를 분리한다.

## 예외

`shared/src/contracts/adapter_contract_families/builtin_loader.py`는 현재 explicit payload family
연결을 유지한다. shared canonical contract는 자동 plugin discovery나 decorator auto registration을
허용하지 않는 방향이므로, 이 파일은 registry smell이 아니라 contract source-of-truth 예외로 본다.
이 규칙을 바꾸려면 contract sync와 golden fixture 검증을 먼저 갱신한다.
