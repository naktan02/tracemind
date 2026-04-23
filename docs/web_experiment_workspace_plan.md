# Web Experiment Workspace Plan

## 1. 목적

이 문서는 `web` 브랜치에서 진행할 개발자용 웹 실험공간 작업의
단계별 실행 계획을 정의한다.

목표는 아래 세 가지다.

1. 현재 TraceMind의 seed, 중앙 적응, FL 실험 축을 웹에서 조합 가능하게 만든다.
2. UI가 source of truth가 되지 않게 유지하면서도 개발자가 새 방법론을
   추가하고 수정하기 쉬운 구조를 만든다.
3. classifier, prototype, PEFT adapter(예: LoRA, DoRA)와 그 조합,
   aggregation, translation 경로를 장기적으로 수용 가능한 형태로 연다.

중요:

- 이 작업은 개발자용 실험 도구 트랙이다.
- 현재 연구/시스템의 canonical 실행 순서
  `seed -> 중앙 LoRA 적응 비교 -> 시스템 FL translation`
  를 뒤집지 않는다.
- 한 번의 패치에서 UI, 계약, runtime, 모든 알고리즘을 동시에 완성하려 들지 않는다.
- future `download/CSV/HF import`는 같은 실험 조합 lane에 섞지 않고,
  별도 `dataset asset lane`으로 분리한다.
  - MVP는 기존 dataset alias만 소비한다.

## 2. 비목표

초기 단계에서 아래는 목표로 잡지 않는다.

1. 모든 논문 방법론을 처음부터 웹에 다 노출
2. 중앙 실험 trainer와 시스템 FL runtime을 한 번에 재설계
3. multi-user production SaaS 수준의 운영 플랫폼 구현
4. 모든 저장/실행을 DB 중심으로 먼저 재구축
5. 현재 scripts/runtime을 즉시 폐기하고 웹만으로 대체

## 현재 상태

기준 시점: `2026-04-23`

이미 완료된 단계:

1. `Phase 0`
   - 계획 문서와 active 진입점을 고정했다.
2. `Phase 1`
   - read-only catalog API를 만들었고,
     track/section/item metadata를 기계적으로 읽을 수 있게 했다.
3. `Phase 2`
   - `WorkspaceManifest`, `ResolvedExperimentPlan`, compile preview API를 만들었다.
   - `core method`, `variant profile`, `override patch` 경계를 manifest에 올렸다.
4. `Phase 2 follow-up cleanup`
   - compiler가 metadata-only block을 조용히 skip하지 않게 막았다.
   - catalog의 compile surface를 명시 계약으로 올렸다.
   - registry-only metadata를 live backend 인스턴스 생성 대신
     source-adjacent catalog entry surface에서 읽게 정리했다.
   - consumer가 없는 package-level barrel export를 정리했다.
5. `Phase 2.5 contract hardening`
   - dataset preset catalog가 asset path, source provenance, readiness를 노출한다.
   - compiler가 `FixMatch`의 unlabeled dataset readiness를 compile 단계에서 검사한다.
   - `federated_run_preset.client_count`를 live roster가 아닌
     synthetic simulation participant count로 명시한다.

현재 남은 단계 우선순위:

1. `Phase 4`
   - local-only 실행 wrapper와 workspace/run 저장
2. `Phase 5`
   - FL baseline workspace와 runtime compile/run
3. `Phase 6`
   - component bundle, translation operator, hybrid path
4. `Phase 7`
   - DoRA/FedMatch/FedRD류 추가 경험 정리

현재 효율성 판단:

1. 지금까지는 비효율적으로 커진 것이 아니라,
   `하드코딩된 추측`, `unused barrel`, `catalog -> live instance coupling`을
   줄이는 방향으로 정리됐다.
2. UI 전에 catalog/manifest/compiler를 먼저 닫은 것은
   이후 화면 변경이 runtime contract를 흔들지 않게 하려는 의도다.
3. registry-only metadata를 source-adjacent surface로 옮긴 것도
   새 backend 추가 시 catalog service까지 매번 뜯지 않게 하려는 정리다.
4. 남아 있는 주의점은 FL block을 너무 빨리 runnable로 열면
   Phase 5 범위가 Phase 3-4를 덮어쓰는 것이다.
   따라서 다음 단계는 local-only 실행/저장을 좁게 시작하고,
   FL 실행 path는 뒤 Phase에서 여는 순서가 맞다.

## 3. 핵심 원칙

1. UI는 source of truth가 아니다.
   - 실제 전략 이름, 계약 의미, 실행 기본값은 코드/contract/Hydra config에 둔다.
2. 단계별로 닫는다.
   - 각 Phase는 독립적인 완료 기준과 커밋 단위를 가진다.
3. 문제 축으로 나눈다.
   - 방법론 이름보다 `component family`, `method`, `objective`,
     `aggregation`, `translation`, `composition`을 먼저 분리한다.
4. track를 섞지 않는다.
   - `seed`, `central adaptation`, `federated runtime`은 다른 workspace lane으로 둔다.
5. 복잡도에 따라 파일과 폴더를 선택한다.
   - 작은 variant는 파일 하나로 두고,
     runner/helper/config/artifact logic가 붙는 큰 family만 폴더로 승격한다.
6. 추가 절차는 통일한다.
   - 구현 추가, registry 등록, metadata 선언, preset 추가, 테스트 추가의
     흐름을 최대한 같은 패턴으로 유지한다.
7. 하이브리드는 나중에 연다.
   - `prototype + classifier`, `peft + classifier` 같은 composition은
     baseline track가 닫힌 뒤에 연다.
8. dataset import와 experiment composition을 분리한다.
   - `download`, `CSV ingest`, `HF import`, path drop은 future dataset asset lane이 맡고,
     workspace lane은 준비된 dataset alias와 asset reference를 소비한다.

## 4. 1급 개념

웹 실험공간은 아래 개념을 직접 표현할 수 있어야 한다.

1. `Track`
   - `seed`, `central_adaptation`, `federated_runtime`
2. `Component Family`
   - `classifier_head`, `prototype_pack`, `peft_adapter`, `diagonal_scale`
3. `Core Method`
   - 구현 레벨의 재사용 가능한 알고리즘 코어
   - 예: `fixmatch`, `lora`, `fedavg`
4. `Method Variant`
   - 예: `lora`, `dora`, `adalora`, `fixmatch`
5. `Variant Profile`
   - 같은 core method를 concrete baseline/ablation으로 고정한 named preset
   - 예: `fixmatch_usb_v1`, `prototype_pseudo_label_v1`
6. `Override Patch`
   - variant profile 위에 run-local로만 덧씌우는 파라미터 변경
   - 예: `temperature=0.7`, `lora.rank=16`
7. `Composition`
   - 예: `head_only`, `prototype_only`, `peft_only`, `peft_plus_head`,
     `prototype_plus_head`
8. `Objective`
   - 예: supervised, pseudo-label self-training, FixMatch, R-Drop
9. `Aggregation Plan`
   - component별 집계 방식
10. `Translation Operator`
   - 예: classifier warm-start, prototype bootstrap, PEFT fallback
11. `Inference Plan`
   - 예: classifier logits, prototype similarity, hybrid fusion
12. `Workspace Manifest`
   - UI 조합 결과를 저장하는 canonical 문서
13. `Compatibility Rule`
    - 무엇이 어떤 track, family, runtime path와 함께 쓸 수 있는지 설명

추가 규칙:

1. `core method`는 가능한 한 코드 구현과 1:1 또는 1:few 관계를 가진다.
2. `variant profile`은 Hydra preset이나 named metadata profile처럼
   사람이 재사용하는 조합 이름이다.
3. 같은 논문 family의 여러 버전은 새 core method를 늘리기보다
   새 variant profile + override patch로 먼저 표현한다.
4. 새로운 수식/로직/학습 경로가 실제로 생길 때만 core method를 늘린다.

## 5. 저장 전략

초기부터 무거운 저장소를 도입하지 않는다.

### Phase 0-3

- catalog, workspace preview, compile 결과는 파일 기반 또는 in-memory로 유지한다.
- artifact는 기존처럼 파일 경로와 manifest를 그대로 재사용한다.

### Phase 4 이후

- 저장이 필요해지면 `SQLite`를 1차 선택지로 둔다.
- 저장 대상은 workspace, run history, 상태 요약, artifact reference다.
- 대용량 checkpoint와 실제 payload는 DB에 넣지 않는다.
- 다중 사용자/원격 협업 요구가 생길 때만 Postgres 계열을 다시 검토한다.

## 6. 디렉터리 전략

초기 목표는 `예쁘게 보이는 구조`가 아니라
`찾기 쉽고 수정 경로가 일관된 구조`다.

권장 방향:

1. `family` 기준으로 1차 분리
2. `method`는 그 아래 배치
3. 작은 variant는 파일 하나
4. 보조 runner, report, config helper가 생기면 폴더로 승격
5. UI 노출 정보는 폴더명이 아니라 metadata에서 읽게 구성

예시 방향:

```text
apps/
  experiment_web/
    src/
      workspace/
      catalog/
      runs/

shared/src/contracts/
  workspace_manifest_contracts.py
  compatibility_contracts.py
  model_bundle_contracts.py

main_server/src/api/
  experiments.py

main_server/src/services/experiments/
  catalog_service.py
  compiler_service.py
  run_service.py

agent/src/services/training/
  query_adaptation/
  components/
    peft_adapter/
      methods/
        lora.py
        dora.py
```

위 예시는 목표 방향을 설명하는 것이며, 초기 Phase에서 모두 만들지 않는다.

## 7. 단계별 계획

### Phase 0. 계획 고정과 용어 정리

상태: 완료

목표:

- web 작업의 범위와 용어를 active doc로 고정한다.

포함:

1. 이 계획 문서 작성
2. active 진입점에서 이 문서를 찾을 수 있게 연결
3. `track`, `component family`, `method`, `composition`, `translation` 용어 고정

제외:

1. UI 구현
2. API 구현
3. 계약 추가

완료 기준:

1. 다음 구현 패치가 이 문서의 Phase 번호를 기준으로 설명될 수 있다.
2. 구현자가 어느 순서로 들어갈지 문서만 보고 이해할 수 있다.

커밋 단위:

- docs only

### Phase 1. Read-only Experiment Catalog

상태: 완료

목표:

- 현재 코드에 이미 있는 전략 축과 preset을 웹이 읽을 수 있는
  machine-readable catalog로 노출한다.

포함:

1. 현재 registry/Hydra config 기반 전략 inventory 정리
2. `seed`, `central_adaptation`, `federated_runtime` track 분리
3. `family`, `core_method`, `variant_profile`, `preset`,
   `supported_runtime_paths` 같은 metadata 정의
4. read-only JSON 또는 API로 catalog 제공
5. registry-only metadata는 live backend 인스턴스 생성 대신 source-adjacent
   catalog entry surface에서 읽는다

제외:

1. workspace 저장
2. 실행
3. 새로운 알고리즘 구현

완료 기준:

1. 웹 또는 CLI에서 현재 선택 가능한 축 목록을 자동으로 읽을 수 있다.
2. `FixMatch`, `classifier_head`, `diagonal_scale`, `fedavg` 같은 기존 축이
   catalog에 명시된다.
3. unsupported 조합과 metadata-only surface가 `compile_support`,
   `compile_blocker_reason`으로 드러난다.

커밋 단위:

- catalog/metadata only

### Phase 2. Workspace Manifest와 Compiler MVP

상태: 완료

목표:

- 사용자가 선택한 블록 조합을 기존 script/runtime이 이해하는 실행 계획으로
  compile하는 계층을 만든다.

포함:

1. `WorkspaceManifest`
2. `ResolvedExperimentPlan`
3. 기존 Hydra/script entrypoint로의 compile 로직
4. `variant profile + override patch -> Hydra override` compile 규칙
5. dry-run preview
6. validation error와 compatibility check
7. entrypoint `script_path`와 compileable selection surface를 catalog metadata로
   직접 읽는다.

제외:

1. 실제 실행
2. DB 저장
3. hybrid multi-component translation

완료 기준:

1. workspace 입력에서 기존 실행 커맨드 또는 override plan이 나온다.
2. 중앙 적응 비교선과 FL baseline이 각각 다른 compile 경로를 가진다.
3. 같은 core method의 여러 variant profile이 manifest 수준에서 구분된다.
4. 잘못된 조합은 compile 단계에서 설명 가능한 오류로 실패한다.

커밋 단위:

- manifest + compiler only

### Phase 3. Web UI MVP

상태: 완료

목표:

- 개발자가 브라우저에서 현재 실험 축을 보고 조합할 수 있는
  read-only 실험공간 UI를 만든다.

포함:

1. palette
2. track별 lane UI
3. block 선택/해제
4. compile preview
5. compatibility 에러 표시
6. `apps/experiment_web` scaffold
7. Vite dev origin용 API CORS 보조
8. catalog-declared `entrypoint_section_name`, `default_slot_name` 기반 UI surface
9. entrypoint별 compile warning/readiness policy registry

제외:

1. 실제 실행
2. run history
3. 다중 사용자 기능

완료 기준:

1. seed 또는 중앙 적응 baseline 하나를 UI에서 조합해 preview할 수 있다.
2. FL baseline은 최소한 read-only 구성 preview가 가능하다.
3. 사용자는 "현재 조합이 어떤 기존 script/config로 번역되는지" 볼 수 있다.
4. UI shell이 `apps/experiment_web`에 격리되고,
   source of truth는 backend/code-adjacent layer에 남는다.
5. UI state와 compiler core가 section 이름/entrypoint 이름 magic string에
   직접 묶이지 않는다.

커밋 단위:

- web UI MVP only

### Phase 4. 실행과 저장 MVP

상태: 대기

목표:

- 승인된 좁은 범위의 실험만 실제로 실행하고 기록할 수 있게 한다.

포함:

1. local-only 실행 wrapper
2. run status 추적
3. workspace 저장/재열기
4. artifact 링크 표시
5. `SQLite` 기반의 run/workspace 메타데이터 저장
6. dataset asset lane과 experiment workspace lane을 저장 계층에서 분리

초기 지원 범위:

1. seed baseline
2. 중앙 supervised LoRA baseline
3. 중앙 pseudo-label self-training
4. 중앙 FixMatch baseline

제외:

1. FL runtime의 모든 경로
2. hybrid composition
3. secure aggregation runtime

완료 기준:

1. UI에서 구성한 중앙 실험을 실제로 시작할 수 있다.
2. run 상태와 artifact 경로를 다시 볼 수 있다.
3. 저장된 workspace를 다시 열어 동일 preview를 확인할 수 있다.

커밋 단위:

- run launcher + local persistence only

### Phase 5. FL Workspace Baseline

상태: 대기

목표:

- 현재 존재하는 시스템 FL baseline을 웹에서 조합 가능하게 만든다.

포함:

1. agent 수
   - 여기서 agent 수는 live server에 등록된 실제 agent roster가 아니라
     `run_federated_simulation`의 synthetic participant count를 뜻한다.
2. dataset/shard policy
3. `adapter_family`
4. `aggregation_backend`
5. `training_algorithm_profile`
6. prototype builder
7. validation/diagnostics 선택

제외:

1. multi-component hybrid aggregation
2. secure aggregation runtime
3. LoRA family FL translation 본 구현

완료 기준:

1. 현재 `run_federated_simulation` baseline을 웹에서 compile/run할 수 있다.
2. `classifier_head`, `diagonal_scale`, `fedavg` 경로를 명시적으로 선택할 수 있다.
3. FL workspace가 중앙 적응 workspace와 섞이지 않는다.
4. `client_count`와 future live roster 개념이 UI/contract에서 분리돼 보인다.

커밋 단위:

- FL baseline workspace only

### Phase 6. Component Bundle과 Translation

상태: 대기

목표:

- classifier, prototype, PEFT adapter를 묶어서 다루는 구조를 연다.

포함:

1. `ModelBundleManifest` 또는 동등한 묶음 계약
2. component별 aggregation plan
3. translation operator registry
4. `prototype -> classifier bootstrap`
5. `classifier -> prototype rebuild`
6. `peft + classifier` bundle 표현

제외:

1. 모든 hybrid 논문 구현
2. 모든 translation operator 구현

완료 기준:

1. 최소 한 개의 hybrid composition이 문서가 아니라 코드 계약으로 표현된다.
2. component별 집계기가 다를 수 있는 구조가 열린다.
3. classifier/prototype/PEFT 간 전환이 implicit가 아니라 명시적 operator로 드러난다.

커밋 단위:

- contract + one translation path + one hybrid path

### Phase 7. 방법론 추가 경험 정리

상태: 대기

목표:

- DoRA, FedMatch, FedRD 같은 방법론을 나중에 넣을 때
  개발자가 헤매지 않도록 추가 절차와 구조를 통일한다.

포함:

1. family/method metadata template
2. registry 추가 패턴 통일
3. 새 방법 추가 플레이북 보강
4. 필요 시 scaffold generator 또는 예제 템플릿

완료 기준:

1. 새 방법 추가 절차가 5단계 내로 설명된다.
2. 작은 variant와 큰 family가 어떤 기준으로 파일/폴더를 쓰는지 명확하다.
3. 한 예시 method를 템플릿처럼 따라 넣을 수 있다.

커밋 단위:

- developer ergonomics only

## 8. Phase Gate

다음 Phase로 넘어가기 전 아래를 확인한다.

1. 이전 Phase 결과가 문서, 테스트, 실행 예시 중 최소 하나로 검증됐다.
2. 새 Phase가 이전 Phase의 source of truth를 덮어쓰지 않는다.
3. 커밋 범위가 하나의 concern으로 설명 가능하다.
4. 사용자가 다음 단계 시작을 명시적으로 승인했다.

## 9. 현재 다음 시작점

다음 구현 시작은 `Phase 4`부터 아래 순서로 잡는다.

1. local-only run launcher 계약을 추가한다.
2. workspace/run 메타데이터 저장을 붙인다.
3. artifact 링크와 상태 요약을 UI에 다시 연결한다.
4. 중앙 baseline 실행 범위를 먼저 좁게 연다.
5. federated runnable path는 `Phase 5`에서 연다.

즉 다음 구현 커밋은 `run launcher + local persistence`가 된다.

## 10. 커밋 원칙

1. 한 커밋은 한 Phase 또는 한 concern만 다룬다.
2. docs-only Phase와 runtime Phase를 섞지 않는다.
3. contract 변경이 있으면 producer, consumer, 테스트, 문서를 같은 흐름에서 닫는다.
4. 웹 UI와 backend compiler를 한 패치에 과도하게 섞지 않는다.
