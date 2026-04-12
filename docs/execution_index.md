# TraceMind Execution Index

## 목적

이 문서는 TraceMind 작업의 짧은 진입점이다.
현재 활성 경로는 `personalized local inference + federated shared model improvement`다.
현재 작업 순서는 `central fixed+classifier seed -> query 적응 LoRA+classifier -> 시스템용 FL translation`으로 본다.

Codex CLI와 VS Code Codex extension을 사용할 때는
`docs/ai_context_manifest.yaml`을 함께 읽어 task별 읽기 순서와
source-of-truth 우선순위를 먼저 확인한다.

## 읽는 순서

1. `docs/ai_context_manifest.yaml`
   - AI용 문맥 지도와 task route
2. `AGENTS.md`
   - 저장소 구조, 소유 경계, 작업 규칙
3. `plan.md` (repo root)
   - 왜 seed 단계와 적응 단계를 분리했는지
4. `docs/project_execution_plan.md`
   - 지금 무엇을 구현하고 무엇을 미루는지, Phase별 현황
5. `shared/src/contracts/README.md`
   - 현재 payload 계약 해석
6. `docs/fl_runtime_implementation_checklist.md`
   - 시스템 FL 트랙의 실제 구현 순서와 완료 기준
7. `docs/contracts/algorithm_extension_guide.md`
   - 알고리즘/전략 교체 시 어느 파일을 보고 어떤 Protocol을 구현할지
8. `docs/contracts/strategy_addition_playbook.md`
   - 새 전략을 어떤 순서로 구현, 등록, 기본값 반영, 테스트할지
9. `docs/strategy_surface_map.md`
   - 지금 실제로 바꿀 수 있는 전략 축과 metadata-only 축을 한눈에 확인
10. `docs/contracts/query_buffer_v1.md`
   - query buffer와 threshold/policy selection의 local boundary
11. 필요한 경우만 `docs/staged_execution_roadmap.md`
   - Phase 이름과 검증 포인트 빠른 확인

## AI Harness 빠른 경로

Codex용 하네스 문서는 아래 순서를 권장한다.

1. `docs/ai_context_manifest.yaml`
2. 관련 path-specific `AGENTS.md`
3. 하네스 자체를 유지보수할 때만 `docs/ai_harness_operating_model.md`
4. harness spot check가 필요할 때만 `docs/ai_harness_eval_cases.yaml`

## 코드 읽기 빠른 경로

문서까지 읽은 뒤 코드를 볼 때는 아래 순서가 가장 빠르다.

### seed / 적응 실험

1. `scripts/README.md`
2. `scripts/experiments/train_softmax_classifier.py`
3. `docs/contracts/query_buffer_v1.md`
4. `docs/contracts/central_lora_classifier_trainer_contract.md`
5. `scripts/experiments/train_lora_classifier.py`
6. 관련 Hydra config (`scripts/conf/experiments/train_lora_classifier.yaml`, `scripts/conf/paper_backbone/*`, `scripts/conf/lora/*`)
7. 필요 시 적응 단계 실험 스크립트/노트북

### agent 로컬 추론/학습

1. `agent/src/services/README.md`
2. `agent/src/services/inference/pipeline_service.py`
3. `agent/src/services/federation/training_example_service.py`
4. `agent/src/services/training/local_training_service.py`
5. `agent/src/services/federation/runtime_service.py`

### main_server FL round orchestration

1. `main_server/src/services/README.md`
2. `main_server/src/services/rounds/README.md`
3. `main_server/src/services/rounds/round_lifecycle_service.py`
4. `main_server/src/services/rounds/round_manager_service.py`

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
| `AGENTS.md` | 저장소 구조, 소유 경계, 작업 규칙 |
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
| `docs/contracts/shared_adapter_contracts_v1.md` | adapter payload 구조와 수학적 의미 |
| `docs/contracts/training_update_envelope_v1.md` | envelope 필드 설계 이유 |

## 작업 시작 체크리스트

1. 이번 요청이 `seed baseline`, `query-domain 적응`, `시스템 FL 트랙` 중 어디인지 먼저 구분한다.
2. 변경 소유 경계가 `shared`, `agent`, `main_server`, `scripts` 중 어디인지 적는다.
3. 바뀔 축이 무엇인지 적는다.
   - 예: classifier family, query buffer, training backend, aggregation backend, privacy layer, scoring policy
4. `docs/contracts/algorithm_extension_guide.md`에서 해당 전략 지점을 찾는다.
5. 운영 후보 로직이면 `scripts`가 아니라 `shared/agent/main_server` 소유 경계에 먼저 둔다.
6. 사용자 판단이 필요한 항목인지 확인한다.

## 문서 우선순위

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `docs/contracts/*`, `docs/*`
4. `docs/notes/*`

## 사용자 확인이 필요한 변경

1. query 버퍼에 raw text를 어떤 retention policy로 남길지
2. 적응 단계 LoRA spec과 threshold/policy를 무엇으로 고정할지
3. `lora` family FL 활성화
4. pseudo-label을 정식 학습 신호로 승격
5. 서버로 보내는 update 메타데이터 확대
6. private adapter/head를 언제 열지
7. secure aggregation, DP, HE 도입 시점
