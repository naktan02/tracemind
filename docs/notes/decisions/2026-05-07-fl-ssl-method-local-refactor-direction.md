# 2026-05-07 FL SSL Method-Local Refactor Direction

이 문서는 다음 세션에서 구조 리팩터링 맥락을 빠르게 복원하기 위한
decision archive다. 현재 source of truth는 아니다. 실제 계약 의미는
`shared/src/contracts/*`, method 구현은 `methods/*`, 실행 조합과 값은 루트
`conf/*` Hydra config가 소유한다.

## 배경

TraceMind는 중앙 SSL pooled/offline control과 FL SSL non-IID main comparison을
모두 실험해야 한다. 앞으로 `FedMatch`, `FedLGMatch`, `(FL)^2` 같은 논문 method를
추가하려면 새 method 하나가 `methods`, `conf`, `scripts`, `agent`,
`main_server`, `shared` 여러 계층을 넓게 흔들지 않아야 한다.

현재 좋은 점도 있다.

1. `methods/ssl`은 이미 `@register_query_ssl_algorithm("fixmatch")`처럼 구현
   파일 옆에서 등록하는 패턴을 일부 갖고 있다.
2. `shared`, `methods`, `agent`, `main_server`, `scripts`의 의존 방향은
   architecture test로 보호된다.
3. `lora_classifier` family는 FL simulation research path에서 먼저 열고,
   live runtime translation은 2차 범위로 두기로 결정돼 있다.

하지만 최근 점검 결과, TraceMind가 USB/SemiLearn보다 길어지는 핵심 원인은
registry보다 아래에 있다. method core가 가져야 할 반복 구조를 `scripts`,
agent backend, runtime adapter, shared contract monolith가 나눠 가지지 못하고
한 파일에 계속 흡수하고 있다.

가장 큰 구조적 마찰은 아래와 같다.

1. `scripts`가 실험 entrypoint가 아니라 algorithm runner 역할을 일부 소유한다.
   예: `scripts/experiments/query_lora_ssl/runners/consistency.py`가
   scripts-local `QuerySslAlgorithmAdapter` registry를 다시 갖고 있다.
2. agent `lora_classifier_trainer.py`는 이름은 trainer지만 현재는 config parsing,
   raw text 추출, label schema resolve, artifact ref 생성, payload 생성,
   client metrics 생성을 모두 처리한다. 실제 LoRA train executor는 아직 없다.
3. fixed classifier와 LoRA classifier가 classification evaluation, best
   checkpoint, epoch history, report scaffold를 반복한다.
4. `PseudoLabelSelectionService.select_evidences()`가 evidence -> candidate,
   hook evaluation, cap/ranking, selection context, feedback 생성을 한 번에
   처리한다.
5. `adapter_contracts.py`와 `training_contracts.py`는 source of truth라는 이유로
   family별 payload, registry, IO, factory까지 한 파일에 모여 있다.
6. `scripts/runtime_adapters/federated_server_runtime.py`는 server runtime build,
   repository wiring, prototype rebuild bridge, initial state factory, round
   request mapper를 한 파일에 갖고 있다.

## USB / SemiLearn 참고

참고 repo:

- <https://github.com/microsoft/Semi-supervised-learning>

중요한 참고 파일:

- `semilearn/core/utils/registry.py`
- `semilearn/core/algorithmbase.py`
- `semilearn/core/hooks/hook.py`
- `semilearn/algorithms/fixmatch/fixmatch.py`
- `semilearn/algorithms/comatch/comatch.py`

USB/SemiLearn이 깔끔해 보이는 핵심은 decorator 문법 하나가 아니다.
`AlgorithmBase`가 train loop, optimizer, eval, checkpoint, hook dispatch를
깊게 흡수하고, 개별 algorithm은 자기 폴더에서 `set_hooks`, `train_step`,
`get_argument`만 채우는 구조가 핵심이다.

예를 들어 FixMatch는 자기 구현 파일에서 algorithm을 등록하고,
`PseudoLabelingHook`과 `FixedThresholdingHook`만 교체한다. CoMatch는 필요한
projection head, memory bank, distribution alignment hook을 자기 algorithm
폴더 주변에 둔다. 이 구조는 caller가 algorithm-specific 세부사항을 넓게 알지
않아도 되게 만드는 깊은 Module이다.

단, USB도 완전 자동 plugin discovery는 아니다. `registry.py`에 `ALL_MODULES`
같은 명시 import 목록이 남아 있다. TraceMind도 이 점을 따른다. 즉,
`decorator 등록 + 명시적 builtin loader import`까지만 가져오고, import order가
불명확한 자동 discovery는 도입하지 않는다.

USB의 algorithm-local `utils.py` 패턴은 그대로 파일명까지 복사하지 않는다.
TraceMind에서는 method 전용 helper를 method 폴더 안에 두되, 관성적인 `utils.py`
보다 `views.py`, `hooks.py`, `losses.py`, `memory_bank.py`, `server_policy.py`
처럼 의미 있는 파일명을 우선한다. 1개 method 전용 helper는 method-local
Module에 두고, 2개 이상 method에서 안정적으로 공유될 때만 `methods/ssl/hooks`
또는 `methods/common`으로 승격한다.

## 좋은 내부 프레임워크 기준

좋은 TraceMind 내부 method framework라면 새 method 추가 시 아래에 가까워야 한다.

1. method-local Module 하나를 추가한다.
2. descriptor/config/profile을 추가한다.
3. 필요한 tensor SSL hook, local update backend, server policy만 교체한다.
4. `scripts`, `agent`, `main_server` core는 거의 건드리지 않는다.
5. 기존 smoke/report/manifest shape는 유지된다.

이번 리팩터링이 끝난다고 바로 "완전한 프레임워크"가 증명되는 것은 아니다.
`fedavg_pseudo_label` 하나만 정리하면 아직 seam은 가설에 가깝다. 실제로
`FedMatch`나 `FedLGMatch` 같은 두 번째, 세 번째 method를 추가하면서
`scripts`/`agent`/`main_server` core 변경이 작게 유지될 때 프레임워크의 Depth가
검증된다.

이번 리팩터링의 목표는 "완성된 범용 프레임워크"가 아니라 "논문 method 추가를
안전하게 반복할 수 있는 내부 도메인 프레임워크의 기반"이다.

프레임워크 v1 foundation으로 인정하려면 구조 분해만으로는 부족하다. 만든 구조가
다시 무너지지 않도록 boundary test, extension test, compatibility validator,
contract golden fixture, 짧은 new method guide가 함께 있어야 한다.

## 결정 내용

1. 첫 vertical slice는 `fedavg_pseudo_label + lora_classifier`로 한다.
   FedMatch나 FedLGMatch를 먼저 새로 만들지 않고, 이미 있는 baseline method를
   method-local 구조의 기준 샘플로 정리한다.

2. FL SSL method identity와 experiment profile을 분리한다.
   `fedavg_pseudo_label`은 method identity다. `lora_classifier`는 adapter/local
   update/round runtime family다. 둘을 묶은 이름은 `method=`가 아니라
   `fl_profile=` 또는 `experiment_profile=` 같은 compose preset으로 표현한다.

3. method-local Module은 `methods/federated_ssl/<method>/`에 둔다.
   현재 repo 경로는 `methods/fl_ssl`이 아니라 `methods/federated_ssl`이므로
   rename은 하지 않는다. `fedavg_pseudo_label/` 안에는 descriptor, required
   views, runtime capability, local objective hint, server policy hint, 짧은
   README를 둔다.

4. method-local default snapshot은 실행값 source of truth가 아니다.
   Python descriptor는 capability와 hint를 소유한다. 실제 backbone, seed,
   LoRA rank/alpha/dropout, client 수, round 수 같은 실행값은 계속 Hydra config가
   소유한다.

5. hook chain은 두 종류를 분리한다.
   `methods/ssl/hooks`는 tensor-level SSL objective hook을 소유한다. 예:
   `MaskingHook`, `PseudoLabelingHook`, `ConsistencyLossHook`, 필요 시
   `DistAlignHook`. 반면 agent의 evidence -> candidate -> feedback 흐름은
   runtime selection/diagnostics policy seam으로 유지한다. 둘을 하나의 hook
   framework로 합치지 않는다.

6. registry 개선은 먼저 `methods`에만 적용한다.
   `methods/ssl`과 `methods/federated_ssl`에 typed registry helper와 decorator
   등록을 적용한다. agent/server registry 통일은 뒤로 미룬다. `shared` payload
   contract와 cross-layer catalog는 명시적 source of truth로 유지한다.

7. `lora_classifier` backend 책임 분리를 앞단으로 당긴다.
   실제 LoRA train executor가 붙기 전에 payload builder, artifact ref,
   row extraction, config parsing, metrics를 분리해야 한다.

8. scripts-local algorithm adapter 중복을 줄인다.
   `methods/ssl` algorithm descriptor가 required views, labeled batch 사용 여부,
   view builder name, algorithm factory를 제공하고, scripts runner는 실행만 맡는
   방향으로 간다.

9. `LocalUpdateProfile` Module은 high-level Hydra profile보다 먼저 만든다.
   trainer, example backend, scorer, evidence, acceptance policy, privacy guard,
   adapter compatibility 조합을 typed profile로 해석할 수 있어야 Hydra compose
   preset도 얇아진다.

10. shared contract monolith 분해는 마지막으로 미룬다.
    `adapter_contracts.py` 분해는 가치가 있지만 첫 vertical slice와 직접 관련이
    약하고 blast radius가 크다. 먼저 parse/serialize 테스트를 보강하고, 후반에
    family별 module + compatibility facade 방식으로 분리한다.

## 왜 이렇게 결정했는가

이 리팩터링의 목적은 코드 줄 수를 줄이는 것이 아니다. 다음 논문 method를 추가할
때 수정 위치가 예측 가능해지는 것이 목적이다.

좋은 구조의 기준:

1. method-specific 지식은 `methods/federated_ssl/<method>/`에 모인다.
2. tensor SSL objective 변경은 `methods/ssl` hook seam에서 끝난다.
3. agent-local privacy와 raw text boundary는 agent runtime에 남는다.
4. server aggregation과 publication mechanism은 main_server runtime adapter에
   남는다.
5. scripts는 Hydra config를 읽고 실행, artifact, report dump만 담당한다.
6. Hydra config는 실행값과 compose preset의 source of truth로 남는다.

USB/SemiLearn에서 가져올 것은 method-local Module, 작은 Interface, hook 교체,
구현 옆 등록, 반복 loop의 deep Interface다. 그대로 가져오지 않을 것은 central
AlgorithmBase가 모든 runtime 경계를 흡수하는 구조다. TraceMind는
privacy/local/server 경계가 더 중요하므로 agent와 main_server runtime을 하나의
trainer base로 합치지 않는다.

## 실행 순서

1. 현재 unit/smoke 기준을 고정한다.
   `resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")`,
   `strategy_axes/fl/method_descriptor=fedavg_pseudo_label`, FL simulation smoke,
   LoRA classifier payload/update manifest, central SSL FixMatch runner, fixed
   classifier seed artifact/report shape가 리팩터링 전후 동일해야 한다.

2. `fedavg_pseudo_label` descriptor를 method-local 구조로 정리한다.
   첫 커밋은 behavior change 없이 descriptor 위치와 소유권만 정리한다.

3. 최소 architecture boundary guard를 추가한다.
   기존 layer dependency test를 유지하되, 리팩터링 초반부터 아래 핵심 규칙은
   깨지지 않게 한다.

   ```text
   shared -> agent/main_server/methods/scripts import 금지
   methods -> scripts import 금지
   ```

   후반에는 더 세밀한 guard로 강화한다.

4. `methods/ssl`과 `methods/federated_ssl`에 typed registry helper를 적용한다.
   자동 discovery가 아니라 명시 builtin loader import를 유지한다.

5. 새 method 추가 비용을 검증하는 extension test를 추가한다.
   `dummy_federated_ssl_method` 또는 아주 작은 toy method를 test-only로 추가해
   method-local Module 추가, builtin loader 한 줄 추가, 필요 시 conf profile
   하나 추가만으로 resolve/compile이 가능한지 검증한다. production registry에
   장난감 method를 영구 추가하지 않는 방향을 우선한다.

6. LoRA classifier backend 책임을 분리한다.
   목표 package:

   ```text
   agent/src/services/training/backends/training/lora_classifier/
     backend.py
     config.py
     row_extractor.py
     label_schema.py
     artifact_refs.py
     payload_builder.py
     metrics.py
     train_executor.py
   ```

   기존 registry name `lora_classifier_trainer`는 유지한다.

7. central SSL runner의 algorithm-specific adapter 중복을 줄인다.
   scripts-local `QuerySslAlgorithmAdapter`를 method descriptor/view spec 쪽으로
   내린다. 단, backtranslation 실행/cache/materialization은 `scripts` 또는
   `scripts/runtime_adapters`가 소유한다.

8. fixed classifier와 LoRA classifier의 classification evaluation, checkpoint,
   epoch history scaffold를 공통화한다.
   후보 package:

   ```text
   methods/adaptation/common/
     classification_evaluation.py
     checkpointing.py
     epoch_history.py
     supervised_trainer.py
   ```

9. SSL tensor hook seam을 실제 중복 기준으로 강화한다.
   FixMatch와 pseudo-label baseline에서 공유되는 것부터 올린다. `DistAlignHook`은
   필요한 method가 생길 때 연다.

10. `LocalUpdateProfile` Module을 도입한다.
   local update 조합과 compatibility를 하나의 deep Module로 검증한다.

11. 최소 FL profile compatibility validator를 추가한다.
    `LocalUpdateProfile`과 `fl_profile`은 조합 자체보다 resolve 시점 검증이
    중요하다. method descriptor, adapter family metadata, local update profile,
    round runtime profile을 함께 검증해 incompatible 조합은 실행 전에 실패하게
    한다.

12. `fl_profile` 또는 `experiment_profile` Hydra compose preset을 추가한다.
   예: `fl_profile=fedavg_pseudo_label_lora_classifier_v1`. 이 이름은 method
   identity가 아니라 실행 조합이다.

13. method-local server/round policy seam을 추가한다.
    FedMatch/FedLGMatch류는 단순 aggregation만 바꾸지 않을 수 있다. client update
    수락 정책, round별 pseudo-label/state exchange, global memory/statistics,
    distillation target, confidence calibration, client weighting이 생길 수 있다.
    따라서 method-local Module에는 필요 시 아래 파일을 둘 수 있게 한다.

    ```text
    methods/federated_ssl/<method>/
      local_objective.py
      server_policy.py
      round_policy.py
    ```

    `fedavg_pseudo_label`은 처음에는 default/empty policy를 써도 된다.

14. server aggregation adapter를 thin화한다.
    aggregation 계산 core는 `methods/federated/aggregation`, runtime payload
    loading, artifact resolve, publication은 `main_server`가 담당한다.

15. simulation server runtime adapter를 package로 분리한다.
    후보 package:

    ```text
    scripts/runtime_adapters/federated_server/
      runtime.py
      repositories.py
      prototype_rebuild_bridge.py
      initial_state_factory.py
      round_request_mapper.py
    ```

16. agent pseudo-label selection pipeline을 분리한다.
    candidate 생성, cap policy, selection context, feedback builder를 별도 Module로
    빼고, `pseudo_label_service.py`는 orchestration만 담당한다.

17. experiment artifact/report writer를 분리한다.
    `SimulationReportBuilder`, `SelectionDiagnosticsWriter`,
    `TeacherPseudoLabelExporter`처럼 산출물별 writer를 둔다.

18. shared contract golden fixture를 추가한다.
    contract facade 분리 전후 shape drift를 막기 위해 golden JSON fixture를 둔다.

    ```text
    tests/contracts/fixtures/lora_classifier_delta.v1.json
    tests/contracts/fixtures/classifier_head_state.v1.json
    tests/contracts/fixtures/training_update_submission.v1.json
    ```

    검증은 load -> parse -> dump -> compare shape, 기존 import path 유지,
    `schema_version` 유지, unknown field 거부 유지까지 포함한다.

19. shared adapter contract family facade 분리를 마지막에 한다.
    기존 import path compatibility와 parse/serialize 테스트를 먼저 보강한다.
    golden fixture를 먼저 만든 뒤 family별 module + compatibility facade 방식으로
    분리한다.

20. new method guide를 추가한다.
    프레임워크 사용법을 짧게 고정한다.

    ```text
    methods/federated_ssl/NEW_METHOD.md
    ```

    내용은 method와 profile 차이, 새 method가 소유해야 할 파일, 건드리면 안 되는
    계층, descriptor 필수 필드, local/server policy 추가 방식, 테스트 체크리스트를
    포함한다.

## 첫 커밋

권장 커밋명:

```text
refactor: fedavg_pseudo_label method descriptor를 method-local 구조로 정리
```

성공 기준:

1. `resolve_federated_ssl_method_descriptor("fedavg_pseudo_label")` 동작 유지.
2. `strategy_axes/fl/method_descriptor=fedavg_pseudo_label` 의미 유지.
3. 관련 unit test 통과.
4. 가능하면 `2 clients / 1 round / 1 seed` FL simulation smoke에서 contract/report
   shape 유지.

## 전체 커밋 순서

1. `test: fedavg_pseudo_label 리팩터링 기준 동작 고정`
2. `refactor: fedavg_pseudo_label method descriptor를 method-local 구조로 정리`
3. `test: fl ssl 최소 architecture boundary guard 추가`
4. `refactor: methods registry helper를 명시적 builtin loader 구조로 정리`
5. `test: federated ssl dummy method extension guard 추가`
6. `refactor: lora classifier backend 책임 분리`
7. `refactor: central ssl runner의 algorithm adapter 중복 제거`
8. `refactor: classifier evaluation과 checkpoint scaffold 공통화`
9. `refactor: ssl tensor hook seam 정리`
10. `refactor: local update profile 조합 module 도입`
11. `feat: fl profile compatibility validator 최소 구현`
12. `feat: fl experiment profile compose preset 추가`
13. `refactor: federated ssl method-local server/round policy seam 추가`
14. `refactor: server aggregation runtime adapter thin화`
15. `refactor: federated simulation server runtime adapter 분리`
16. `refactor: pseudo label selection pipeline 책임 분리`
17. `refactor: experiment artifact writer 책임 분리`
18. `test: shared contract golden fixture 추가`
19. `refactor: shared adapter contract family facade 분리`
20. `docs: federated ssl new method guide 추가`

## 다음 세션 시작점

다음 세션에서 바로 시작하려면 아래 순서로 읽는다.

1. `AGENTS.md`
2. `docs/ai_context_manifest.yaml`
3. `docs/execution_index.md`
4. `docs/project_execution_plan.md`
5. 이 문서
6. `methods/README.md`
7. `methods/federated_ssl/registry.py`
8. `methods/federated_ssl/fedavg_pseudo_label/*`
9. `agent/src/services/training/backends/training/lora_classifier_trainer.py`
10. `scripts/experiments/query_lora_ssl/runners/consistency.py`
11. `methods/adaptation/lora_classifier/training.py`
12. `scripts/experiments/fixed_classifier/runner.py`
13. `tests/unit/test_scripts_hydra_configs.py`
14. `tests/architecture/test_layer_dependencies.py`
15. `tests/contracts/fixtures/*`
16. `methods/federated_ssl/NEW_METHOD.md`

첫 작업은 기준 테스트 확인과 `fedavg_pseudo_label` method-local descriptor 정리다.
FedMatch, FedLGMatch, `(FL)^2` 신규 구현은 이 vertical slice가 안정화된 뒤에
추가한다.

## 제외 범위

1. experiment workspace catalog/UI 전면 정리.
2. `methods/federated_ssl`를 `methods/fl_ssl`로 rename.
3. 자동 plugin discovery.
4. shared payload contract 자동 decorator 등록.
5. FedMatch/FedLGMatch 신규 구현 선행.
6. agent와 main_server runtime을 하나의 trainer base로 합치기.
