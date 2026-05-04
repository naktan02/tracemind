# Agent Services

이 디렉터리는 agent가 소유하는 로컬 runtime 서비스 모음이다.

핵심 원칙:

- `shared` 계약을 읽어 local inference와 local training을 실행한다.
- 서버와 통신하는 orchestration은 `federation/`에 두고,
  실제 scoring/training 메커니즘은 `inference/`, `training/`에 둔다.
- asset cache/sync와 language helper는 각각 `assets/`, `language/`로 분리한다.
- `services/*/__init__.py`는 기본적으로 marker만 두고 direct-file import를 우선한다.
- newcomer는 `__init__.py`보다 아래 읽기 순서로 시작하는 편이 빠르다.

## 하위 패키지 역할

- `inference/`
  - 로컬 추론 rail
  - global classifier/prototype evidence 계산, 의사결정, 시계열 누적 담당
- `training/`
  - 로컬 학습 rail
  - selection, example assembly, execution, dataset 조립, backend 구현 담당
- `federation/`
  - agent와 서버 사이 round orchestration rail
  - current round fetch/upload 담당
- `assets/prototypes/`
  - prototype artifact 동기화/로컬 runtime helper
- `language/`
  - preprocess, translation, backtranslation helper
- `wellbeing/`
  - 가족용 확장 프로그램이 읽는 전체 상태/추이/잠금용 local output service

## 먼저 읽을 파일

현재 v1에서 권장하는 읽기 관점:

- `global classifier`는 공통 evidence producer다.
- `local interpretation`이 final decision owner다.
- shared adapter와 prototype scoring은 비교/확장 경로로 유지한다.

### 1. 로컬 추론 흐름을 보고 싶을 때

1. `inference/pipeline_service.py`
2. `inference/scoring_service.py`
3. `inference/decision_service.py`
4. `inference/time_series_service.py`
5. `language/translation_service.py`

### 2. 로컬 학습 흐름을 보고 싶을 때

1. `training/examples/service.py`
2. `training/selection/pseudo_label_service.py`
3. `training/execution/runtime_compatibility.py`
4. `training/execution/local_training_service.py`
5. `training/backends/`

### 3. agent가 서버 round에 참여하는 흐름을 보고 싶을 때

1. `federation/rounds/round_client.py`
2. `federation/rounds/runtime_service.py`
3. `training/execution/local_training_service.py`

### 4. 가족용 확장 출력 surface를 보고 싶을 때

1. `wellbeing/family_access_service.py`
2. `wellbeing/summary_service.py`
3. `wellbeing/timeseries_service.py`
4. `wellbeing/projection_service.py`
5. `wellbeing/auth_service.py`
6. `wellbeing/child_support_service.py`
7. `wellbeing/child_support_context_provider.py`
8. `wellbeing/child_support_safety_policy.py`
9. `wellbeing/child_support_llm_provider.py`
10. `agent/src/api/family_access.py`
11. `agent/src/api/child_support.py`
12. `agent/src/api/wellbeing.py`

## 파일 역할 빠른 맵

- `inference/scoring_backends.py`
  - scorer backend registry와 agent runtime adapter 구현
- `methods/prototype/scoring/`
  - prototype similarity와 score 집계 policy core
- `training/selection/pseudo_label_service.py`
  - score를 pseudo-label candidate/accepted set으로 해석
- `docs/contracts/query_buffer_v1.md`
  - agent-owned local query retention과 selection 입력 경계
- `infrastructure/repositories/query_buffer_repository.py`
  - raw text + prediction snapshot을 로컬에 저장하는 query buffer 저장소
- `training/selection/query_buffer_projection.py`
  - query buffer snapshot + scored event를 `methods/prototype/evidence/` core로
    `PseudoLabelEvidence`에 투영
- `training/selection/query_buffer_selection_service.py`
  - query buffer 기반 selection runner
- `training/selection/query_buffer_selection_diagnostics.py`
  - selection 결과를 family-agnostic summary/trace 진단 shape로 정리
- `training/selection/pseudo_label_service.py`
  - `methods/ssl/hooks/`의 selection hook을 agent-local
    candidate/context/diagnostics로 감싼다
- `language/backtranslation_service.py`
  - 운영 translation 코어와 같은 층에서 재사용하는 backtranslation service
  - strict USB NLP input용 `aug_0`, `aug_1` strong candidate 생성에 재사용한다
- `training/datasets/query_adaptation_dataset_service.py`
  - accepted pseudo-label candidate를 raw-text adaptation dataset으로 조립
  - `query_id`는 `source_row.query_id`를 single source of truth로 두고,
    locale/source_type/model_revision은 typed provenance로 보존
- `training/datasets/query_adaptation_multiview_service.py`
  - single-view adaptation dataset을 weak/strong source row가 있는 multiview dataset으로 확장
  - augmentation recipe는 여기서 고정하지 않고 pluggable augmenter hook으로 분리
- `training/selection/query_buffer_lifecycle_service.py`
  - query buffer raw text retention / purge 정책과 lifecycle 실행
- `training/examples/models.py`
  - local training과 federation이 공유하는 example DTO
- `training/execution/runtime_compatibility.py`
  - training/example/scorer/privacy 조합 검증
- `training/examples/service.py`
  - raw row 또는 stored event를 `EmbeddedTrainingExample`으로 변환
- `training/backends/inputs/`
  - single-view, weak/strong pair 같은 training input backend 구현
- `training/backends/evidence/`
  - pseudo-label evidence 정규화 backend 구현
- `training/acceptance_policies/`
  - evidence 기반 pseudo-label acceptance 정책 구현
- `training/backends/training/`
  - adapter update 생성 backend adapter와 registry wiring
- `methods/adaptation/diagonal_scale/`
  - diagonal-scale local update 계산 core

## 전략 추가 시 출발점

- training backend 추가: `methods/adaptation/<family>/`와 `training/backends/training/`
- example-generation backend 추가: `training/backends/inputs/`와 `training/examples/service.py`
- scorer backend 추가: `inference/scoring_backends.py`
- prototype score policy 추가: `methods/prototype/scoring/`
- privacy guard 추가: `training/execution/privacy_guard_service.py`

확장 전에 `shared/src/contracts/README.md`,
`docs/contracts/algorithm_extension_guide.md`,
`docs/contracts/strategy_addition_playbook.md`를 함께 읽는 것을 권장한다.
