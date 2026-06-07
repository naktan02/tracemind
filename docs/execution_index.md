# TraceMind Execution Index

짧은 문서 지도다. 세부 계약 의미는 코드와 code-adjacent README를 우선한다.

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
| `docs/project_execution_plan.md` | 현재 활성 판단과 다음 검증 |
| `docs/architecture/system-overview.md` | 런타임 구성요소와 코드 경계 |
| `docs/architecture/target-method-runtime-structure.md` | 최종 method/runtime 구조 |
| `shared/src/contracts/README.md` | payload 계약 해석 |
| `methods/README.md` | 재사용 algorithm/method core 소유 경계 |
| `conf/README.md` | Hydra config group과 실행 축 |
| `scripts/README.md` | 실험 entrypoint와 workflow 경계 |
| `docs/operations/local-runbook.md` | 로컬 실행, GPU preflight, smoke |
| `docs/quality/test-strategy.md` | 테스트 층과 보호 범위 |
| `docs/governance/document-governance.md` | 문서 class와 갱신 규칙 |

## Contract And Product Docs

| 문서 | 역할 |
|---|---|
| `docs/contracts/query_buffer_v1.md` | query retention과 selection boundary |
| `docs/contracts/central_peft_text_encoder_trainer_contract.md` | 중앙 SSL control scaffold |
| `docs/contracts/legacy_contract_ledger.md` | legacy/compatibility 표면의 소유자와 제거 조건 |
| `docs/api/api-surface.md` | FastAPI endpoint 표면과 owner |
| `agent/src/contracts/README.md` | agent-local API/UI payload 계약 해석 |
| `apps/family_extension/README.md` | family extension 현재 UI/runtime 표면 |

## Fast Code Paths

Central SSL:

1. relevant `conf/**`
2. `scripts/experiments/central/ssl_control/README.md`
3. `scripts/experiments/central/ssl_control/run_peft_supervised_control.py`
4. `scripts/experiments/central/ssl_control/run_peft_ssl_control.py`
5. `scripts/support/query_ssl_text_encoder/runners/**`
6. `docs/contracts/central_peft_text_encoder_trainer_contract.md`
7. `methods/ssl/**`, `methods/adaptation/{peft_text_encoder,full_text_encoder}/**`

FL SSL:

1. relevant `conf/**`
2. `scripts/experiments/fl_ssl/README.md`
3. `scripts/experiments/fl_ssl/run_federated_simulation.py`
4. `scripts/experiments/fl_ssl/federated_simulation/**`
5. `scripts/runtime_adapters/federated_{agent,server}/**`
6. `methods/federated_ssl/**`
7. `methods/adaptation/peft_text_encoder/**`

Agent runtime:

1. `agent/AGENTS.md`
2. `agent/src/services/README.md`
3. `agent/src/services/inference/**`
4. `agent/src/services/training/**`
5. `agent/src/services/federation/**`

Main server FL:

1. `main_server/AGENTS.md`
2. `main_server/src/services/README.md`
3. `main_server/src/services/federation/rounds/README.md`
4. `main_server/src/services/federation/aggregation/**`

## Start Checklist

1. 요청이 seed, central SSL control, FL SSL non-IID, runtime translation 중 어디인지 구분한다.
2. 변경 소유 경계가 `shared`, `methods`, `conf`, `agent`, `main_server`, `scripts`,
   `apps`, `docs` 중 어디인지 정한다.
3. 전략/알고리즘 추가라면 `methods/README.md`, 해당 `methods/**/NEW_METHOD.md`,
   `conf/README.md`를 먼저 본다.
4. SSL 논문 비교라면 중앙 control과 FL main comparison을 분리한다.
5. `docs/notes/**`는 archive-only로 둔다.
6. FL SSL smoke/main/sweep은 기본적으로 `gpu_local + mxbai`로 실행한다.
   `gpu_online`은 cache warm-up/최초 다운로드용이고, `cpu_local + hash_debug`는
   entrypoint wiring smoke나 빠른 단위 검증에만 쓴다.
