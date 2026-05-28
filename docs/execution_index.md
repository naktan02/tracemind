# TraceMind Execution Index

짧은 문서 지도다. task별 read order는 `docs/ai_context_manifest.yaml`을 우선한다.

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

## Core Docs

| 문서 | 역할 |
|---|---|
| `AGENTS.md` | repo-wide 작업 규칙과 소유 경계 |
| `docs/ai_context_manifest.yaml` | task route와 source-of-truth 우선순위 |
| `plan.md` | 연구 비전과 global/local 경계 |
| `docs/project_execution_plan.md` | active decision, phase, next step |
| `docs/architecture/system-overview.md` | 런타임 구성요소와 코드 경계 |
| `docs/architecture/target-method-runtime-structure.md` | 최종 method/runtime 구조와 migration plan |
| `docs/architecture/method-owned-runtime-refactor-plan.md` | method-owned core와 runtime adapter 경계 guard |
| `docs/architecture/pattern-integrity-refactor-backlog.md` | 패턴 경계 guard와 남은 후보 |
| `docs/architecture/code-expression-guidelines.md` | 코드 표현 밀도와 읽기 난이도 기준 |
| `shared/src/contracts/README.md` | payload 계약 해석 |
| `docs/contracts/legacy_contract_ledger.md` | legacy/compatibility 표면의 소유자와 제거 조건 |
| `docs/operations/local-runbook.md` | 로컬 실행, GPU preflight, smoke |
| `docs/quality/test-strategy.md` | 테스트 층과 보호 범위 |
| `docs/governance/document-governance.md` | 문서 class와 갱신 규칙 |
| `methods/README.md` | 재사용 algorithm/method core 소유 경계 |
| `methods/evaluation/README.md` | 중앙/FL 공통 evaluation metric helper 경계 |
| `conf/README.md` | Hydra config group과 strategy/runtime 축 |

## Research Docs

| 문서 | 역할 |
|---|---|
| `docs/contracts/query_buffer_v1.md` | query retention과 selection boundary |
| `docs/contracts/central_peft_classifier_trainer_contract.md` | 중앙 SSL control scaffold |
| `docs/fl_runtime_implementation_checklist.md` | FL/runtime translation 작업표 |
| `docs/strategy_surface_map.md` | 전략 축, 기본값, 구현 상태 |
| `docs/contracts/strategy_addition_playbook.md` | 새 strategy 추가 절차 |
| `docs/contracts/algorithm_extension_guide.md` | 새 protocol/전략 축 세부 |
| `docs/contracts/fl_ssl_method_capability_matrix.md` | FedMatch/FedLGMatch/(FL)^2 선택 전 capability matrix |

## Reference / Historical Docs

| 문서 | 역할 |
|---|---|
| `docs/notes/decisions/2026-05-28-archived-text-classifier-adaptation-refactor-plan.md` | 완료된 text classifier migration 기록. legacy shim 변경 때만 읽는다 |
| `docs/notes/decisions/2026-05-28-archived-lora-classifier-v1-terminology-audit.md` | 완료된 `lora_classifier` 용어 감사 기록. legacy migration 추적 때만 읽는다 |
| `docs/notes/decisions/2026-05-28-archived-strategy-surface-map.md` | 긴 strategy surface 감사 기록. 현재 축 확인은 active `docs/strategy_surface_map.md`를 먼저 읽는다 |
| `docs/notes/decisions/2026-05-28-archived-fl-ssl-runbook.md` | 긴 FL SSL 실행 예시 기록. 현재 실행 경계는 `scripts/experiments/fl_ssl/README.md`를 먼저 읽는다 |
| `docs/notes/decisions/2026-05-28-archived-fl-simulation-package-readme.md` | 긴 FL simulation package 설명 기록. 현재 패키지 경계는 code-adjacent README를 먼저 읽는다 |
| `docs/notes/sessions/2026-05-28-archived-fl-ssl-execution-audit.md` | 특정 시점 FL SSL artifact 감사 기록 |
| `docs/notes/sessions/2026-05-28-archived-fl-ssl-runtime-performance-audit.md` | FedMatch runtime 성능 개선 전후 수치 기록 |
| `docs/staged_execution_roadmap.md` | phase 이름만 보는 축약 지도. 현재 우선순위는 project execution plan이 소유 |

## Fast Code Paths

Seed / central SSL:

1. relevant `conf/**`
2. `docs/architecture/target-method-runtime-structure.md`
3. `docs/contracts/query_buffer_v1.md`
4. `docs/contracts/central_peft_classifier_trainer_contract.md`
5. `scripts/experiments/central_ssl_control/train_peft_supervised_classifier.py`
6. `scripts/experiments/central_ssl_control/train_peft_ssl_classifier.py`
7. `scripts/experiments/query_peft_ssl/*`
8. `methods/adaptation/peft_text_encoder/*`
9. `methods/evaluation/*`
10. `methods/adaptation/query_text_views/*`
11. `methods/ssl/NEW_METHOD.md` (새 Query SSL algorithm 추가 시)
12. `methods/ssl/*`
13. `methods/adaptation/*`

Agent runtime:

1. `agent/src/services/README.md`
2. `docs/architecture/method-owned-runtime-refactor-plan.md`
3. `agent/src/services/inference/pipeline_service.py`
4. `agent/src/services/training/execution/agent_training_task_runner_service.py`
5. `agent/src/services/training/execution/local_training_service.py`
6. `agent/src/services/federation/rounds/runtime_service.py`

Main server FL:

1. `main_server/src/services/README.md`
2. `docs/architecture/method-owned-runtime-refactor-plan.md`
3. `main_server/src/services/federation/rounds/round_lifecycle_service.py`
4. `main_server/src/services/federation/rounds/round_manager_service.py`

Apps:

1. `apps/AGENTS.md`
2. app-specific `AGENTS.md`
3. `shared/src/contracts/README.md`

## Start Checklist

1. 요청이 seed, central SSL control, FL SSL non-IID, runtime translation 중 어디인지 구분한다.
2. 변경 소유 경계가 `shared`, `methods`, `conf`, `agent`, `main_server`, `scripts`, `apps`, `docs` 중 어디인지 정한다.
3. 전략/알고리즘 추가라면 `docs/architecture/target-method-runtime-structure.md`를
   먼저 보고, 현행 실행 표면은 `docs/strategy_surface_map.md`에서 확인한다.
4. SSL 논문 비교라면 중앙 control과 FL main comparison을 분리한다.
5. `docs/notes/**`는 archive-only로 둔다.
6. FL SSL smoke/main/sweep은 기본적으로 `gpu_local + mxbai`로 실행한다.
   `gpu_online`은 cache warm-up/최초 다운로드용이고, `cpu_local + hash_debug`는
   entrypoint wiring smoke나 빠른 단위 검증에만 쓴다.
