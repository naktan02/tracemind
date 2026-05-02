# TraceMind Execution Index

## 목적

이 문서는 TraceMind 작업의 짧은 진입점이다.
현재 활성 경로는 `personalized local inference + federated shared model improvement`다.
현재 작업 순서는 `central fixed+classifier seed -> query continual adaptation LoRA+classifier -> 시스템용 FL translation`으로 본다.

Codex CLI와 VS Code Codex extension을 사용할 때는
`docs/ai_context_manifest.yaml`의 task route를 먼저 보고, 아래 문서 지도에서
필요한 문서만 고른다. 아래 목록은 전체 읽기 순서가 아니다.

## 문서 지도

- `docs/ai_context_manifest.yaml`
  - AI용 문맥 지도와 task route
- `AGENTS.md`
  - 저장소 구조, 소유 경계, 작업 규칙
- `docs/architecture/system-overview.md`
  - 현재 런타임 구성요소, 활성 rail, 코드 소유 경계
- `docs/operations/local-runbook.md`
  - 로컬 실행, GPU preflight, smoke 절차
- `plan.md` (repo root)
  - 왜 seed 단계와 적응 단계를 분리했는지
- `docs/project_execution_plan.md`
  - 지금 무엇을 구현하고 무엇을 미루는지, Phase별 현황
- `docs/api/api-surface.md`
  - agent/main_server API route 표면과 contract source
- `shared/src/contracts/README.md`
  - 현재 payload 계약 해석
- `docs/fl_runtime_implementation_checklist.md`
  - 시스템 FL 트랙의 실제 구현 순서와 완료 기준
- `docs/strategy_surface_map.md`
  - 지금 실제로 바꿀 수 있는 전략 축과 metadata-only 축을 한눈에 확인
- `docs/contracts/strategy_addition_playbook.md`
  - 새 전략을 어떤 순서로 구현, 등록, 기본값 반영, 테스트할지
- `docs/contracts/algorithm_extension_guide.md`
  - 새 Protocol이나 전략 축 구현 세부가 필요할 때만 본다
- `docs/contracts/query_buffer_v1.md`
  - query buffer와 threshold/policy selection의 local boundary
- `shared/src/contracts/workspace_manifest_contracts.py`
  - Phase 2 workspace manifest와 compile preview canonical contract
- `apps/AGENTS.md`
  - web/app 구현을 볼 때만 읽는다
- `apps/experiment_web/AGENTS.md`
  - 실험 workspace UI 구현을 볼 때만 읽는다
- `docs/staged_execution_roadmap.md`
  - Phase 이름과 검증 포인트가 필요할 때만 읽는다
- `docs/family_extension_wellbeing_signal_mvp_plan.md`
  - 가족용 확장 MVP를 볼 때만 읽는다
- `apps/family_extension/AGENTS.md`
  - 가족용 확장 UI 구현을 볼 때만 읽는다

## AI Harness 빠른 경로

Codex용 하네스 문서는 아래 순서를 권장한다.

1. `docs/ai_context_manifest.yaml`
2. 관련 path-specific `AGENTS.md`
3. 실험 의도 정렬이나 skill 선택이 애매할 때만
   `.codex/skills/tracemind-research-loop/SKILL.md`
4. 하네스 자체를 유지보수할 때만 `docs/ai_harness_operating_model.md`
5. harness spot check가 필요할 때만 `docs/ai_harness_eval_cases.yaml`

## 코드 읽기 빠른 경로

필요한 문서만 확인한 뒤 코드를 볼 때는 아래 순서가 가장 빠르다.

### seed / 적응 실험

1. 관련 Hydra config (`scripts/conf/experiments/train_lora_classifier.yaml`, `scripts/conf/experiments/train_lora_bootstrap_classifier_teacher.yaml`, `scripts/conf/experiments/train_lora_pseudo_label_classifier.yaml`, `scripts/conf/experiments/train_lora_fixmatch.yaml`, `scripts/conf/query_adaptation_initial_checkpoint/*`, `scripts/conf/query_ssl_method/*`, `scripts/conf/query_ssl_train_source/*`, `scripts/conf/query_ssl_augmenter/*`, `scripts/conf/paper_backbone/*`, `scripts/conf/lora/*`)
2. `scripts/experiments/train_softmax_classifier.py`
3. `docs/contracts/query_buffer_v1.md`
4. `docs/contracts/central_lora_classifier_trainer_contract.md`
5. `scripts/experiments/train_lora_classifier.py`
6. `scripts/experiments/train_lora_bootstrap_classifier_teacher.py`
7. `scripts/experiments/train_lora_pseudo_label_classifier.py`
8. `scripts/experiments/lora_classifier/runner.py`
9. `scripts/experiments/lora_classifier/bootstrap_runner.py`
10. `scripts/experiments/lora_classifier/query_adaptation_runner.py`
11. `scripts/experiments/lora_classifier/pseudo_label_runner.py`
12. `scripts/experiments/lora_classifier/query_ssl/consistency_runner.py`
13. `scripts/experiments/lora_classifier/query_ssl/augmentation.py`
14. `scripts/README.md`는 command discovery나 cross-script convention이 필요할 때만 읽는다.
15. 필요 시 적응 단계 실험 스크립트/노트북

### agent 로컬 추론/학습

1. `agent/src/services/README.md`
2. `agent/src/services/inference/pipeline_service.py`
3. `agent/src/services/training/examples/service.py`
4. `agent/src/services/training/execution/local_training_service.py`
5. `agent/src/services/federation/rounds/runtime_service.py`

### main_server FL round orchestration

1. `main_server/src/services/README.md`
2. `main_server/src/services/federation/rounds/README.md`
3. `main_server/src/services/federation/rounds/round_lifecycle_service.py`
4. `main_server/src/services/federation/rounds/round_manager_service.py`

### 실험 entrypoint

1. `scripts/experiments/README.md`
2. `scripts/experiments/prototype_strategy/README.md`
3. `scripts/experiments/federated_simulation/README.md`

### prototype artifact build/eval

1. `scripts/prototypes/README.md`
2. `scripts/prototypes/seed_prototypes.py`
3. `scripts/prototypes/evaluate_prototype_pack.py`

## 문서 역할

| 파일 | 역할 |
|---|---|
| `docs/ai_context_manifest.yaml` | AI용 문맥 지도, task route, source-of-truth 우선순위 |
| `docs/ai_harness_operating_model.md` | 하네스 유지보수용 보조 문서 |
| `docs/ai_harness_eval_cases.yaml` | maintainer 전용 harness spot check sample |
| `.codex/skills/tracemind-research-loop/SKILL.md` | 실험 의도 정렬과 project-local skill routing |
| `.codex/skills/THIRD_PARTY_NOTICES.md` | 외부에서 가져온 project-local skills 출처와 license |
| `AGENTS.md` | 저장소 구조, 소유 경계, 작업 규칙 |
| `docs/architecture/system-overview.md` | 현재 런타임, 활성 rail, 코드 소유 경계 |
| `docs/api/api-surface.md` | agent/main_server API route 표면과 contract source |
| `docs/operations/local-runbook.md` | 로컬 실행, GPU preflight, smoke 절차 |
| `docs/quality/test-strategy.md` | 테스트 층과 보호 범위 |
| `docs/governance/document-governance.md` | 문서 class, source-of-truth, 갱신 규칙 |
| `plan.md` | 연구 비전, 핵심 가설, staged seed/adaptation 원칙 |
| `docs/project_execution_plan.md` | 활성 아키텍처, 현재 Phase, 다음 액션, 검증 기준 |
| `docs/fl_runtime_implementation_checklist.md` | 시스템 FL 트랙 구현 작업표 |
| `docs/staged_execution_roadmap.md` | 중복 설명 없이 Phase map만 제공 |
| `shared/src/contracts/README.md` | 코드 가까운 contract 해설 |
| `docs/contracts/` | 계약 설계 배경, 알고리즘 확장 가이드 |
| `docs/contracts/algorithm_extension_guide.md` | 교체 가능한 모든 전략 지점과 교체 절차 |
| `docs/contracts/central_lora_classifier_trainer_contract.md` | query-domain LoRA 적응 scaffold와 산출물 경계 |
| `docs/contracts/query_buffer_v1.md` | agent-owned local query retention, selection 입력, adaptation 준비 경계 |
| `docs/contracts/strategy_addition_playbook.md` | 전략 추가 시 실제 작업 순서와 검증 순서 |
| `docs/strategy_surface_map.md` | 전략 축, 기본값 source, 실험 override 가능 여부, 구현 상태 |
| `docs/family_extension_wellbeing_signal_mvp_plan.md` | 아이용/부모용 확장과 wellbeing signal 출력 계약 기반 MVP 계획 |
| `shared/src/contracts/workspace_manifest_contracts.py` | workspace manifest, core method/variant profile/override patch, compile preview 계약 |
| `docs/contracts/shared_adapter_contracts_v1.md` | adapter payload 구조와 수학적 의미 |
| `docs/contracts/training_update_envelope_v1.md` | envelope 필드 설계 이유 |

## 작업 시작 체크리스트

1. 이번 요청이 `seed baseline`, `query-domain 적응`, `시스템 FL 트랙` 중 어디인지 먼저 구분한다.
2. 변경 소유 경계가 `shared`, `agent`, `main_server`, `scripts` 중 어디인지 적는다.
3. 바뀔 축이 무엇인지 적는다.
   - 예: classifier family, query buffer, training backend, aggregation backend, privacy layer, scoring policy
4. 전략/알고리즘 추가라면 `docs/strategy_surface_map.md`에서 현재 활성 축인지 먼저 확인한다.
5. 새 Protocol이나 구현 세부가 필요할 때만 `docs/contracts/algorithm_extension_guide.md`를 본다.
6. 운영 후보 로직이면 `scripts`가 아니라 `shared/agent/main_server` 소유 경계에 먼저 둔다.
7. 사용자 판단이 필요한 항목인지 확인한다.

## 운영 문서 빠른 경로

1. 시스템 구조 확인: `docs/architecture/system-overview.md`
2. API route 확인: `docs/api/api-surface.md`
3. 로컬 실행과 smoke: `docs/operations/local-runbook.md`
4. 테스트 범위 확인: `docs/quality/test-strategy.md`
5. 문서 갱신 기준 확인: `docs/governance/document-governance.md`

## 문서 우선순위

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `docs/contracts/*`, active `docs/*.md`, `docs/architecture/*`, `docs/api/*`,
   `docs/operations/*`, `docs/quality/*`, `docs/governance/*`

`docs/notes/**`는 archive-only다. 현재 규칙으로 쓰려면 active docs나
code-adjacent 문서로 요약 승격한 뒤 참조한다.

## 사용자 확인이 필요한 변경

1. query 버퍼에 raw text를 어떤 retention policy로 남길지
2. 적응 단계 LoRA spec과 threshold/policy를 무엇으로 고정할지
3. `lora` family FL 활성화
4. pseudo-label을 정식 학습 신호로 승격
   - central canonical 비교는 seed 1회 이후 new accepted query-derived rows only continual adaptation을 기준으로 본다.
5. 서버로 보내는 update 메타데이터 확대
6. private adapter/head를 언제 열지
7. secure aggregation, DP, HE 도입 시점
