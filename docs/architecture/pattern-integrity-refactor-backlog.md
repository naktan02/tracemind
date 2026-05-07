# Pattern Integrity Refactor Backlog

이 문서는 패턴 이름은 붙어 있지만 책임 경계가 아직 충분히 깊지 않은 후보를 추적한다.
source of truth는 코드와 code-adjacent README이며, 이 문서는 다음 리팩터링 순서를 정하는
작업 목록이다.

## 2026-05-08 닫은 항목

- `methods/federated/aggregation/registry.py`: registry primitive에서 FedAvg concrete import와
  하단 등록 block을 제거했다. 각 FedAvg core function이 decorator로 method metadata를
  등록한다. 남은 import trigger cleanup은 별도 backlog에서 추적한다.
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
  제거하고 shared adapter family metadata + aggregation backend로 조합되는 generic
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

## 2026-05-08 진행 중인 Batch 1 항목

- `agent/src/services/training/acceptance_policies/`
  - 방향: acceptance policy를 SSL selection hook과 중복 판단하지 않는 metadata/compatibility
    seam으로 낮춘다.
  - 완료 기준: `registry.py`는 primitive, `top1.py`는 implementation-local catalog와
    decorator registration, selection 판단은 `methods/ssl/hooks/selection.py`가 맡는다.
- `agent/src/services/training/backends/evidence/`
  - 방향: evidence backend 구현 옆 factory/catalog 등록, registry primitive,
    resolver 분리.
- `agent/src/services/training/backends/inputs/`
  - 방향: input backend 구현 옆 factory/catalog 등록, registry primitive,
    compatibility/resolver 분리.
- `agent/src/services/inference/scoring_backends/`
  - 방향: monolith module을 package로 나누고 concrete scoring adapter는 구현 옆 decorator로
    등록한다.
- `agent/src/services/training/backends/training/`
  - 완료: concrete local update backend를 제거하고 registry facade만 남긴다.

## Registry Backlog

- Import trigger cleanup
  - 문제: `builtin_loader.py`는 registry 하단 등록 block보다 낫지만, concrete module 목록을
    중앙 파일에 옮긴 것만으로는 최종 구조가 아니다.
  - 방향: name-to-module convention, config-declared module path, package manifest 중
    해당 축에 맞는 방식을 선택한다. shared canonical contract family만 explicit list 예외다.
  - 남은 대상: 새로 발견되는 일반 runtime/strategy registry. 현재 Batch 1.5 대상은 닫힘.
- `methods/prototype/scoring/policies.py`
  - 문제: score policy 구현과 registry가 같은 파일에 있고 하단 등록 block을 쓴다.
  - 방향: policy 구현 옆 decorator 등록 또는 `policies/` package로 분리한다.
- `main_server/src/services/experiment_workspace/compiler/policies.py`
  - 문제: compile policy 구현, registry, default registry instance가 한 파일에 모여 있다.
  - 방향: policy 구현과 registry primitive를 분리하고 policy import trigger는
    convention/config 기반으로 둔다.

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
  - 문제: adapter family별 aggregation adapter는 현재 얇지만, FedMatch/FedLGMatch server-side
    policy가 들어오면 round lifecycle이나 aggregation adapter가 method 의미를 흡수할 수 있다.
  - 방향: method-specific server/round policy는 `methods/federated_ssl/<method>/`가 소유하고,
    `main_server`에는 generic server policy executor와 round state exchange capability만 둔다.
- `main_server/src/services/experiment_workspace/catalog/local_training_registry_snapshot.py`
  - 패턴: Catalog/Facade
  - 진행: shared가 agent local training implementation module path를 들고 있던
    catalog snapshot을 main_server experiment workspace catalog로 이동했다.
  - 남은 문제: workspace UI/catalog 노출용 stable snapshot이므로 implementation
    registration source of truth가 되면 안 된다.
  - 방향: agent runtime registry wiring은 구현 옆 catalog entry를 사용한다. main_server
    snapshot은 read-only display facade로만 유지하고 새 method/experiment-only backend를
    여기에 누적하지 않는다.
- `methods/federated_ssl/training_algorithm_profiles.py`,
  `training_default_values.py`, `training_defaults.py`
  - 패턴: Profile / Defaults
  - 진행: shared에서 제거해 FL SSL method/profile 계층인 `methods/federated_ssl`로
    이동했다.
  - 남은 문제: Hydra local update profile과 Python mapping/default가 같은 실행 조합 의미를
    반복할 수 있다.
  - 방향: 실행값은 Hydra config에 두고, Python module은 compatibility facade,
    typed parser/validator, 테스트 fixture helper 중 하나로 역할을 좁힌다.
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
- `main_server/src/services/experiment_workspace/compiler/policies.py`
  - 패턴: Policy
  - 문제: compile policy가 판단뿐 아니라 config patch 조립과 compile output shape 일부를 함께 만든다.
  - 방향: policy는 선택/허용 규칙을 맡고 patch builder가 config mutation을 맡는다.
- `scripts/experiments/prototype_analysis/prototype_strategy/sweep.py`
  - 패턴: Runner/Policy
  - 문제: threshold policy 실행, evaluation 선택, artifact write가 runner에 같이 있다.
  - 방향: experiment runner는 orchestration만 맡기고 evaluator, selection policy, writer를 분리한다.

## 예외

`shared/src/contracts/adapter_contract_families/builtin_loader.py`는 현재 explicit payload family
연결을 유지한다. shared canonical contract는 자동 plugin discovery나 decorator auto registration을
허용하지 않는 방향이므로, 이 파일은 registry smell이 아니라 contract source-of-truth 예외로 본다.
이 규칙을 바꾸려면 contract sync와 golden fixture 검증을 먼저 갱신한다.
