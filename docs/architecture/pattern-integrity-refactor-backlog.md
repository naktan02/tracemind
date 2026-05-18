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
  concrete backend import와 하단 등록 block을 제거한 뒤, old path compatibility facade도
  삭제했다. concrete backend 구현과 registry는 `methods/adaptation/`이 소유한다.
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
- `methods/adaptation/lora_classifier/update/local_update.py`: LoRA-classifier local update의
  training row, label schema resolve, train executor contract, shared delta payload
  조립을 method-local core로 이동했다. `agent` lora backend는 raw text 추출과
  agent-local artifact ref glue를 methods core에 연결한다.
- `agent/src/services/training/execution/privacy_guards/`: privacy guard Protocol,
  concrete guard, registry, catalog entry를 한 파일에서 분리했다. registry는
  `RuntimeRegistry` primitive를 쓰고 guard 구현 옆 decorator가 factory/catalog를
  등록한다.
- `methods/federated/aggregation/fedavg/strategy.py`: FedAvg method 선택과 공통
  lineage/type 검증을 담당하는 generic strategy wiring으로 제한했다.
- `methods/adaptation/<family>/aggregation/fedavg.py`: adapter family별 FedAvg core,
  payload projection, 다음 state materialization을 family-owned module로 이동했다.
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
- research web backend와 UI는 더 이상 활성 런타임이 아니어서 제거했다.

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
  - 완료: concrete local update backend와 old path registry facade를 모두 제거했다.

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
4. Batch 6: scripts runtime adapter, artifact writer/exporter, diagnostics serializer
   후보는 1차 완료됐다. 새 후보가 발견되면 runner가 orchestration만 맡는지와
   단순 facade가 생기지 않는지를 architecture guard로 먼저 닫는다.
5. Batch 7: dummy method dry-run과 extension guard로 새 method 추가 비용을 검증한다.

## 다음 Profile 단계 체크리스트

다음 profile 단계는 패턴 정리가 아니라 실행 조합의 source-of-truth를 줄이는 작업이다.
목표는 Hydra config가 실행값을 소유하고, Python은 조합 해석과 실패 시점 검증만 맡게
만드는 것이다.

- `conf/strategy_axes/fl/**`와 `methods/federated_ssl/*profile*.py`,
  `methods/federated_ssl/training_default*.py`의 중복 실행값을 표로 만든다.
- threshold, scorer, evidence backend, privacy guard, local update backend,
  adapter family, aggregation backend의 canonical owner를 하나씩 정한다.
- Python module이 실행 기본값을 만들고 있으면 validator, resolver, legacy facade,
  test fixture helper 중 하나로 역할을 낮춘다.
- 완료: compatibility validator는 method descriptor와 recipe metadata,
  LocalUpdateProfile, adapter family, aggregation backend를 한 번에 본다.
- 완료: round runtime/objective drift 검증은
  `methods/adaptation/runtime_objective_compatibility.py` dispatcher가 adapter kind별
  구현 옆 `runtime_compatibility.py`를 lazy import하는 구조로 정리했다.
- 완료: high-level compose preset을 제거하고 실행값 source-of-truth를
  `method_descriptor`, `local_update_profile`, `round_runtime.*` leaf로 단일화했다.
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
## Non-Registry Pattern Backlog

- `agent/src/services/training/backends/training/`
  - 패턴: Runtime Adapter / Port
  - 진행: `LocalUpdateExecutor` port가 update submission 실행을 맡도록 분리했다.
  - 진행: LoRA-classifier label schema/train executor/payload stats core를 methods로
    이동했다.
  - 진행: diagonal-scale과 LoRA-classifier concrete training backend를
    `methods/adaptation/<family>/training_backend.py`로 이동했다.
  - 남은 방향: `agent`는 raw text/private state/artifact/payload upload capability만
    맡고, 새 backend 추가 시 `agent/src/services/training/backends/training/` path를
    재도입하지 않는다.
- `main_server/src/services/federation/rounds/`
  - 패턴: Runtime Adapter / Server Policy
  - 진행: adapter family별 round family 파일은 generic runtime으로 접었다.
  - 진행: FedAvg method 선택과 공통 lineage/type 검증은
    `methods/federated/aggregation/fedavg/strategy.py`로 이동했다.
  - 진행: adapter family별 FedAvg core와 payload projection은
    `methods/adaptation/<family>/aggregation/fedavg.py`로 이동했다.
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
  - 완료: `SimulationReportBuilder`는 payload dict만 만들고,
    `SimulationReportWriter`가 report path와 JSON serialization을 맡는다.
  - 완료 기준: builder가 directory 생성, JSON serialization, 파일 write를 직접 소유하지
    않고 architecture guard가 회귀를 막는다.
- `scripts/experiments/query_lora_ssl/io/teacher_pseudo_label_exporter.py`
  - 패턴: Builder/Exporter
  - 문제: teacher evidence/row 변환과 prediction artifact 쓰기가 같은 exporter에 있다.
  - 완료: `TeacherPseudoLabelBuilder`는 pseudo-label row, prediction trace, summary payload만
    만들고 `TeacherPseudoLabelArtifactWriter`가 JSON/JSONL 저장을 맡는다.
  - 완료 기준: legacy `teacher_pseudo_label_exporter.py` facade를 제거하고 architecture
    guard가 재도입을 막는다.
- `scripts/experiments/query_lora_ssl/io/artifacts.py`
  - 패턴: Writer/Exporter
  - 문제: run directory, adapter save, tokenizer save, classifier manifest, report write가 한 함수에
    모여 있다.
  - 완료: `write_run_artifacts()`는 public orchestration entrypoint로 유지하고,
    경로 계산(`artifact_paths.py`), 모델 export(`model_artifact_exporter.py`),
    payload build(`manifest_builder.py`), JSON write(`artifact_writer.py`)로 분리했다.
  - 완료 기준: `artifacts.py`가 JSON serialization, model save, 파일 write를 직접 소유하지
    않고 architecture guard가 회귀를 막는다.
- `scripts/runtime_adapters/federated_agent_runtime.py`
  - 패턴: Runtime Adapter
  - 문제: script runtime mapping, weak/strong row validation, training task 생성, backend resolve가
    한 adapter에 남아 있다.
  - 완료: monolith `federated_agent_runtime.py`를 제거하고 `federated_agent/` package로
    분리했다. scoring, selection, training request bridge, training example mapper,
    row validator, backend resolver를 별도 module이 소유한다.
  - 완료 기준: caller는 역할별 module을 직접 import하고 architecture guard가 legacy monolith
    재도입을 막는다.
- `scripts/experiments/prototype_analysis/prototype_strategy/sweep.py`
  - 패턴: Runner/Policy
  - 문제: threshold policy 실행, evaluation 선택, artifact write가 runner에 같이 있다.
  - 완료: threshold policy 후보 평가(`threshold_policy_evaluator.py`), selection policy
    (`threshold_selection.py`), artifact write(`threshold_artifact_writer.py`)를 분리했다.
  - 완료 기준: `sweep.py` runner는 embedding/strategy/scorer 조합과 summary assembly만
    맡고, policy 후보 평가 loop와 JSON 저장은 전용 module이 맡는다.

## Code Expression Simplification Track

목표는 현재의 `methods`/`agent`/`main_server`/`shared` 경계를 유지하면서, 읽는 표면을
일반적인 Python 오픈소스에 가깝게 낮추는 것이다. 구조 자체를 되돌리지는 않는다.
`shared` contract, cross-layer payload, runtime seam에는 강한 타입을 남기고, runner,
catalog, diagnostics, rule table, fallback/profile 선언부는 가능한 한 평범한 함수,
dict/list table, 작은 helper로 낮춘다.

진행 규칙:

- 한 배치는 한 concern만 고치고, 검증 후 커밋/푸시한다.
- 동작 변경 없이 표현 밀도부터 낮춘다. contract 의미 변경은 별도 contract sync로 다룬다.
- 단일 사용처용 `Protocol`, frozen `dataclass`, compatibility facade, wrapper class는 삭제
  테스트로 먼저 검증한다.
- 새 method 추가 위치는 계속 `methods/`와 `conf/`가 기본이고, `agent`/`main_server`는
  capability adapter만 소유한다.

배치:

- E0. 용어/주석 품질 정리
  - 상태: 완료.
  - 대상: 깨진 한국어 주석과 active backlog의 단계 기준.
  - 완료 기준: 계약 의미와 payload shape 변경 없이 주석/계획만 정리한다.
- E1. `scripts` runner 표현 단순화
  - 상태: 완료.
  - 대상: `scripts/experiments/fixed_classifier/runner.py`,
    `scripts/experiments/query_lora_ssl/runners/*`.
  - 진행: fixed classifier의 data model과 artifact write/load를
    `scripts/experiments/fixed_classifier/models.py`, `artifacts.py`로 분리했다. 기존
    `runner.py` public import 표면은 유지한다.
  - 진행: fixed classifier prediction loop를 `prediction.py`로, row embedding batching을
    `row_embeddings.py`로 분리했다. bootstrap runner는 prediction helper를 직접 import한다.
  - 진행: fixed classifier 평가 계산/출력을 `evaluation.py`로 분리했다. runner는 평가 실행
    시점과 결과 저장만 맡는다.
  - 진행: fixed classifier label tensor 변환과 classifier head 학습 loop를 `training.py`로
    분리했다. runner는 embedding 준비, training helper 호출, 평가 orchestration만 맡는다.
  - 진행: bootstrap teacher runner의 seed/unlabeled row resolve와 split manifest write를
    `teacher_split.py`로 분리했다.
  - 진행: bootstrap teacher runner의 teacher artifact 재사용/학습/저장을
    `teacher_classifier.py`로 분리했다.
  - 진행: pseudo-label self-training runner의 labeled row JSONL/manifest/summary writer를
    `io/labeled_row_export.py`로 분리했다.
  - 진행: pseudo-label self-training runner의 input source resolve와 query_id 검증을
    `pseudo_label_inputs.py`로 분리했다.
  - 완료 판단: `fixed_classifier/runner.py`는 158 lines, `bootstrap_teacher.py`는 199 lines,
    `pseudo_label.py`는 203 lines로 낮아졌다. `consistency.py`, `query_adaptation.py`,
    `supervised.py`는 core training, IO, context build를 이미 별도 module에 위임하고 있어
    이번 배치에서는 추가 분리하지 않는다.
  - 방향: entrypoint는 config load, core 호출, artifact 저장 호출만 남기고 학습/평가/예측
    세부 loop는 작은 module 함수로 분리한다. `Any`는 외부 library boundary 또는 Hydra raw
    config boundary에만 남긴다.
- E2. `agent` diagnostics/rule table 단순화
  - 상태: 완료.
  - 대상: `agent/src/services/training/selection/query_buffer_selection_diagnostics.py`,
    wellbeing rule table 후보.
  - 진행: query buffer selection diagnostics의 row/summary/scalar stats 전용 DTO와
    dataclass serializer helper를 제거했다. service는 JSON-ready dict payload를 만들고,
    reporting writer는 해당 payload를 그대로 기록한다.
  - 진행: child-support response policy의 긴 `build_plan()` 분기를 plan table lookup
    흐름으로 낮췄다. 안전 문구, fallback text, required/blocked term 검증 의미는 유지한다.
  - 방향: stable contract가 아닌 diagnostics는 class graph보다 plain dict builder와
    serialization helper를 우선한다. runtime adapter는 private state, artifact ref, capability
    호출만 맡는다.
- E3. `methods` descriptor/config/fallback 표현 축소
  - 상태: 완료.
  - 대상: `methods/federated_ssl/base.py`, `runtime_fallbacks.py`,
    `methods/adaptation/lora_classifier/config.py`.
  - 진행: `runtime_fallbacks.py`의 반복 objective getter와 fallback mapping merge 코드를
    작은 helper로 낮췄다. fallback 값의 source-of-truth와 profile dataclass 의미는 유지한다.
  - 진행: `methods/adaptation/lora_classifier/config.py`의 필드별 `from_mapping()` 파싱을
    config key table 기반 loop로 낮췄다. backend config dataclass와 값 검증 의미는 유지한다.
  - 진행: `methods/federated_ssl/base.py`의 descriptor dataclass 반복 검증 setter를
    파일 내부 helper로 낮췄다. descriptor/recipe/capability 계약 surface는 유지한다.
  - 방향: method core의 tensor/objective 계산은 유지하고, descriptor/config/fallback 선언부에서
    단일 구현 전용 추상화와 장황한 generic type을 줄인다.
- E5. `shared` contract 파일 크기와 설명 품질 정리
  - 상태: 완료.
  - 대상: `shared/src/contracts/training_contracts.py`를 비롯한 큰 contract 파일.
  - 진행: `TrainingObjectiveConfigPayload`와 `TrainingSelectionPolicyPayload`를
    `shared/src/contracts/training_objective_contracts.py`로 분리했다. 기존
    `training_contracts.py` import path는 compatibility surface로 유지한다.
  - 진행: `SecureAggregationConfigPayload`와 `SecureAggregationSubmissionPayload`를
    `shared/src/contracts/secure_aggregation_contracts.py`로 분리했다. task/update
    envelope의 기존 import path는 compatibility surface로 유지한다.
  - 점검: `shared/src/contracts`에서 300줄을 넘는 파일은 `training_contracts.py`와
    `training_objective_contracts.py`뿐이며, 깨진 한국어 주석 잔재는 발견되지 않았다.
  - 방향: 타입 엄격성은 유지하되 파일을 의미 단위로 나누고, 필드 의미는 코드 가까이에 짧고
    정확한 한국어로 남긴다.

## 예외

`shared/src/contracts/adapter_contract_families/builtin_loader.py`는 현재 explicit payload family
연결을 유지한다. shared canonical contract는 자동 plugin discovery나 decorator auto registration을
허용하지 않는 방향이므로, 이 파일은 registry smell이 아니라 contract source-of-truth 예외로 본다.
이 규칙을 바꾸려면 contract sync와 golden fixture 검증을 먼저 갱신한다.
