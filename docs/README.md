# TraceMind Docs

이 디렉터리는 TraceMind를 처음 읽는 사람과 개발자가 필요한 문서를 빠르게 고르기 위한
문서 진입점입니다. 세부 payload field의 source of truth는 문서가 아니라
`shared/src/contracts/*.py`, `agent/src/contracts/*.py`, code-adjacent README입니다.

## Start Here

| Goal | Read |
|---|---|
| 프로젝트 전체 구조를 보고 싶다 | [architecture/system-overview.md](architecture/system-overview.md) |
| 로컬에서 실행하고 싶다 | [operations/local-runbook.md](operations/local-runbook.md) |
| API 표면을 보고 싶다 | [api/api-surface.md](api/api-surface.md) |
| 테스트 범위를 알고 싶다 | [quality/test-strategy.md](quality/test-strategy.md) |
| 문서 운영 규칙을 보고 싶다 | [governance/document-governance.md](governance/document-governance.md) |

## By Area

### Experiments

- [../scripts/experiments/central/ssl_control/README.md](../scripts/experiments/central/ssl_control/README.md)
  - 중앙 SSL control, PEFT/full text encoder 지도학습 baseline 실행 예시를 설명합니다.
- [../scripts/experiments/central/fixed_feature_control/README.md](../scripts/experiments/central/fixed_feature_control/README.md)
  - fixed-feature 지도학습 baseline과 classical self-training 실행 예시를 설명합니다.
- [../scripts/experiments/fl_ssl/README.md](../scripts/experiments/fl_ssl/README.md)
  - FL SSL simulation, materialized split, method-owned/manual 조합 예시를 설명합니다.

### Architecture

- [architecture/system-overview.md](architecture/system-overview.md)
  - 현재 runtime 구성, 주요 rail, 코드 소유 경계를 설명합니다.
- [architecture/target-method-runtime-structure.md](architecture/target-method-runtime-structure.md)
  - method/runtime 경계의 목표 구조와 migration 방향을 설명합니다.
- [architecture/live-fssl-runtime-translation.md](architecture/live-fssl-runtime-translation.md)
  - FL SSL simulation 구조를 live runtime으로 옮길 때의 지도입니다.

### Operations

- [operations/local-runbook.md](operations/local-runbook.md)
  - 설치, API 서버, frontend, dashboard, GPU preflight, smoke 절차를 정리합니다.

### Contracts

- [contracts/central_peft_text_encoder_trainer_contract.md](contracts/central_peft_text_encoder_trainer_contract.md)
  - 중앙 query-domain PEFT text encoder trainer와 artifact 경계를 설명합니다.
- [contracts/model_manifest_v1.md](contracts/model_manifest_v1.md)
  - model manifest 계약을 설명합니다.
- [contracts/training_task_v1.md](contracts/training_task_v1.md)
  - training task payload 계약을 설명합니다.
- [contracts/training_update_envelope_v1.md](contracts/training_update_envelope_v1.md)
  - training update envelope 계약을 설명합니다.
- [contracts/legacy_contract_ledger.md](contracts/legacy_contract_ledger.md)
  - legacy/compatibility 표면의 소유자와 제거 조건을 정리합니다.

### API

- [api/api-surface.md](api/api-surface.md)
  - agent와 main_server의 FastAPI route 표면과 owner를 요약합니다.

### Quality

- [quality/test-strategy.md](quality/test-strategy.md)
  - unit, integration, e2e, architecture guard가 각각 무엇을 보호하는지 설명합니다.

### Governance

- [governance/document-governance.md](governance/document-governance.md)
  - active docs, code-adjacent docs, archive notes의 역할을 구분합니다.

## Code-Adjacent Guides

상세 구현을 읽을 때는 가까운 README를 우선합니다.

| Area | Guide |
|---|---|
| Shared contracts | [../shared/src/contracts/README.md](../shared/src/contracts/README.md) |
| Methods | [../methods/README.md](../methods/README.md) |
| Hydra config | [../conf/README.md](../conf/README.md) |
| Scripts | [../scripts/README.md](../scripts/README.md) |
| Agent | [../agent/README.md](../agent/README.md) |
| Main server | [../main_server/README.md](../main_server/README.md) |
| Family extension | [../apps/family_extension/README.md](../apps/family_extension/README.md) |
| Experiment dashboard | [../apps/experiment_dashboard/README.md](../apps/experiment_dashboard/README.md) |

## Maintainer References

- [execution_index.md](execution_index.md)
  - 내부 작업자와 AI agent가 task route를 고를 때 쓰는 빠른 지도입니다.
- [ai_context_manifest.yaml](ai_context_manifest.yaml)
  - source-of-truth 우선순위와 task route를 담은 machine-friendly manifest입니다.
- [notes/README.md](notes/README.md)
  - archive-only notes 위치입니다. 현재 규칙이나 절차의 source of truth로 쓰지 않습니다.
