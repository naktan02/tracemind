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
| `shared/src/contracts/README.md` | payload 계약 해석 |
| `docs/operations/local-runbook.md` | 로컬 실행, GPU preflight, smoke |
| `docs/quality/test-strategy.md` | 테스트 층과 보호 범위 |
| `docs/governance/document-governance.md` | 문서 class와 갱신 규칙 |

## Research Docs

| 문서 | 역할 |
|---|---|
| `docs/contracts/query_buffer_v1.md` | query retention과 selection boundary |
| `docs/contracts/central_lora_classifier_trainer_contract.md` | 중앙 SSL control scaffold |
| `docs/fl_runtime_implementation_checklist.md` | FL/runtime translation 작업표 |
| `docs/staged_execution_roadmap.md` | Phase map과 current checkpoint |
| `docs/strategy_surface_map.md` | 전략 축, 기본값, 구현 상태 |
| `docs/contracts/strategy_addition_playbook.md` | 새 strategy 추가 절차 |
| `docs/contracts/algorithm_extension_guide.md` | 새 protocol/전략 축 세부 |

## Fast Code Paths

Seed / central SSL:

1. relevant `scripts/conf/**`
2. `docs/contracts/query_buffer_v1.md`
3. `docs/contracts/central_lora_classifier_trainer_contract.md`
4. `scripts/experiments/train_lora_classifier.py`
5. `scripts/experiments/train_lora_pseudo_label_classifier.py`
6. `scripts/experiments/train_lora_fixmatch.py`
7. `agent/src/services/training/query_classifier_adaptation/*`
8. `agent/src/services/training/query_ssl_algorithms/*`
9. `agent/src/services/training/peft_adapters/*`

Agent runtime:

1. `agent/src/services/README.md`
2. `agent/src/services/inference/pipeline_service.py`
3. `agent/src/services/training/execution/local_training_service.py`
4. `agent/src/services/federation/rounds/runtime_service.py`

Main server FL:

1. `main_server/src/services/README.md`
2. `main_server/src/services/federation/rounds/round_lifecycle_service.py`
3. `main_server/src/services/federation/rounds/round_manager_service.py`

Apps:

1. `apps/AGENTS.md`
2. app-specific `AGENTS.md`
3. `shared/src/contracts/README.md`

## Start Checklist

1. 요청이 seed, central SSL control, FL SSL non-IID, runtime translation 중 어디인지 구분한다.
2. 변경 소유 경계가 `shared`, `agent`, `main_server`, `scripts`, `apps`, `docs` 중 어디인지 정한다.
3. 전략/알고리즘 추가라면 `docs/strategy_surface_map.md`를 먼저 확인한다.
4. SSL 논문 비교라면 중앙 control과 FL main comparison을 분리한다.
5. `docs/notes/**`는 archive-only로 둔다.
