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
  concrete backend import와 하단 등록 block을 제거했다. backend factory 등록은 구현 module이
  소유한다.
- `methods/ssl/hooks/registry.py`: selection hook registry를 lookup/decorator primitive로
  낮췄다. 남은 import trigger cleanup은 별도 backlog에서 추적한다.
- `main_server/src/services/federation/rounds/families/registry.py`: registry 하단
  concrete family builder와 등록 block을 제거했다. 추가로 concrete 목록을 가진
  `builtin_loader.py` 대신 family name -> module name convention 기반 lazy import로
  바꿨다.
- `agent` local runtime registries: training/evidence/input/acceptance/scoring registry에서
  concrete 목록을 가진 `builtin_loader.py`를 제거했다. package-local naming convention
  import helper로 필요한 module만 import하고, catalog listing은 package-local module을
  bounded import한다.

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
  - 방향: agent training backend를 method identity가 아니라 local runtime capability adapter로
    문서화한다. 다음 batch에서 methods-owned local update executor port로 더 얇게 만든다.

## Registry Backlog

- Import trigger cleanup
  - 문제: `builtin_loader.py`는 registry 하단 등록 block보다 낫지만, concrete module 목록을
    중앙 파일에 옮긴 것만으로는 최종 구조가 아니다.
  - 방향: name-to-module convention, config-declared module path, package manifest 중
    해당 축에 맞는 방식을 선택한다. shared canonical contract family만 explicit list 예외다.
  - 남은 대상: `main_server` aggregation registry, `methods/federated/aggregation`,
    `methods/ssl/hooks`.
- `agent/src/services/training/execution/privacy_guard_service.py`
  - 문제: privacy guard Protocol, 구현체, registry, service helper가 한 파일에 섞여 있다.
  - 방향: `privacy_guards/` package에서 base, concrete guards, registry를 분리하고,
    import trigger는 concrete 목록 loader가 아니라 convention/config 기반으로 둔다.
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
  - 문제: adapter family별 local update runtime adapter가 계속 늘어나면 `agent`가 method
    framework 역할을 흡수할 수 있다. 현재 diagonal-scale과 LoRA classifier wrapper는 methods
    core를 호출하지만, 장기적으로는 새 method 추가 때 agent 파일이 늘어나는 흐름을 막아야 한다.
  - 방향: `methods/adaptation/<family>/`가 local update core와 method/family 의미를 소유하고,
    `agent`는 raw text/private state/artifact/payload upload capability adapter로 얇게 둔다.
- `main_server/src/services/federation/rounds/`
  - 패턴: Runtime Adapter / Server Policy
  - 문제: adapter family별 aggregation adapter는 현재 얇지만, FedMatch/FedLGMatch server-side
    policy가 들어오면 round lifecycle이나 aggregation adapter가 method 의미를 흡수할 수 있다.
  - 방향: method-specific server/round policy는 `methods/federated_ssl/<method>/`가 소유하고,
    `main_server`에는 generic server policy executor와 round state exchange capability만 둔다.
- `shared/src/config/local_training_registry_catalog.py`
  - 패턴: Catalog/Facade
  - 문제: shared가 agent local training implementation module path와 catalog entry를
    중앙 파일 하나에 모아 둔다. 구현 옆 registry metadata와 cross-layer workspace catalog
    snapshot의 source가 섞인다.
  - 방향: `RegistryCatalogEntry` 타입과 catalog payload contract만 shared에 남기고,
    agent runtime registry wiring은 구현 옆 catalog entry를 사용한다. main_server workspace
    catalog가 agent를 직접 import할 수는 없으므로, shared 파일은 별도 catalog snapshot 또는
    compatibility facade로 낮춘 뒤 제거 기준을 둔다.
- `shared/src/config/training_algorithm_profiles.py`,
  `training_default_values.py`, `training_defaults.py`
  - 패턴: Profile / Defaults
  - 문제: Hydra local update profile과 Python mapping/default가 같은 실행 조합 의미를
    반복할 수 있다. 이 상태가 커지면 실행값 source of truth가 `conf/`와 `shared/src/config`
    사이에서 갈라진다.
  - 방향: 실행값은 Hydra config에 두고, shared Python module은 stable contract constant,
    compatibility facade, typed parser/validator 중 하나로 역할을 좁힌다.
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
