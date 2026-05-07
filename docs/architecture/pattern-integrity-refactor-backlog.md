# Pattern Integrity Refactor Backlog

이 문서는 패턴 이름은 붙어 있지만 책임 경계가 아직 충분히 깊지 않은 후보를 추적한다.
source of truth는 코드와 code-adjacent README이며, 이 문서는 다음 리팩터링 순서를 정하는
작업 목록이다.

## 2026-05-08 닫은 항목

- `methods/federated/aggregation/registry.py`: registry primitive에서 FedAvg concrete import와
  하단 등록 block을 제거했다. 각 FedAvg core function이 decorator로 method metadata를
  등록하고 `builtin_loader.py`는 명시 import만 한다.
- `main_server/src/services/federation/rounds/aggregation/registry.py`: runtime aggregation
  backend registry를 primitive로 낮췄다. backend factory 등록은 각 backend module 옆 decorator가
  소유한다.
- `agent/src/services/training/backends/training/registry.py`: training backend registry에서
  concrete backend import와 하단 등록 block을 제거했다. backend factory 등록은 구현 module이
  소유한다.
- `methods/ssl/hooks/registry.py`: selection hook registry를 lookup/decorator primitive로
  낮췄다. builtin hook module은 explicit import loader로만 불러온다.

## Registry Backlog

- `agent/src/services/training/acceptance_policies/registry.py`
  - 문제: registry가 concrete acceptance policy import와 등록 block을 함께 소유한다.
  - 방향: `top1.py`가 decorator로 policy factory를 등록하고 registry는 lookup/catalog만 맡긴다.
- `agent/src/services/training/backends/evidence/registry.py`
  - 문제: evidence backend registry가 concrete backend와 default resolve 흐름을 함께 안고 있다.
  - 방향: backend 구현 옆 등록, registry primitive, resolve helper 분리.
- `agent/src/services/training/backends/inputs/registry.py`
  - 문제: example backend registry가 concrete backend import, compatibility check, training backend
    resolve를 함께 처리한다.
  - 방향: registry primitive, `compatibility.py`, implementation-local registration으로 분리.
- `agent/src/services/inference/scoring_backends.py`
  - 문제: Protocol, concrete scoring backends, registry, backend-name helper가 한 파일에 있다.
  - 방향: `scoring_backends/` package로 쪼개고 backend 구현 옆 decorator 등록을 적용한다.
- `agent/src/services/training/execution/privacy_guard_service.py`
  - 문제: privacy guard Protocol, 구현체, registry, service helper가 한 파일에 섞여 있다.
  - 방향: `privacy_guards/` package에서 base, concrete guards, registry, builtin loader를 분리한다.
- `main_server/src/services/federation/rounds/families/registry.py`
  - 문제: round family registry가 concrete family class와 builder function을 직접 소유한다.
  - 방향: family module이 factory 등록을 소유하고 registry는 family lookup만 맡긴다.
- `methods/prototype/scoring/policies.py`
  - 문제: score policy 구현과 registry가 같은 파일에 있고 하단 등록 block을 쓴다.
  - 방향: policy 구현 옆 decorator 등록 또는 `policies/` package로 분리한다.
- `main_server/src/services/experiment_workspace/compiler/policies.py`
  - 문제: compile policy 구현, registry, default registry instance가 한 파일에 모여 있다.
  - 방향: policy 구현과 registry primitive를 분리하고 builtin policy loader를 둔다.

## Non-Registry Pattern Backlog

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
