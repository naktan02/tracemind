# Legacy Contract Ledger

이 문서는 아직 제거하지 않은 legacy/compatibility 표면의 소유자와 제거 조건을
기록한다. 필드 의미의 source of truth는 여전히 code-adjacent contract와 runtime
파일이며, 이 문서는 리팩터링 전 baseline과 제거 판단 기준만 보조한다.

## Baseline Surface

리팩터링 전후 호환성을 볼 때 최소한 아래 표면은 유지돼야 한다.

| 표면 | 기준 파일 | 권장 검증 |
|---|---|---|
| shared contract parse/serialize | `shared/src/contracts/*.py` | `shared/tests/unit/*contracts*` |
| Hydra/script entrypoint import | `conf/**`, `scripts/**` | `tests/architecture/test_script_entrypoint_imports.py` |
| server/agent boundary | `main_server/src/**`, `agent/src/**`, `shared/src/contracts/**` | 관련 service unit, integration/e2e |
| report/dashboard schema | `scripts/experiments/**/export*`, `apps/experiment_dashboard/**` | result index/dashboard fixture smoke |
| architecture guard | `tests/architecture/**` | layer dependency, import guard |

노트북 환경에서는 CUDA 의존 smoke를 실행하지 않는다. 빠른 정적 검증은
`py_compile`, import search, whitespace diff check로 닫고, GPU/pytest full suite는
실제 실행 환경에서 별도 수행한다.

## Active Legacy Entries

| Legacy 표면 | Canonical 표면 | 소유자 | 현재 소비자 | 유지 이유 | 제거 조건 |
|---|---|---|---|---|---|
| `TrainingTaskPayload.secure_aggregation_required` input key | `TrainingTaskPayload.secure_aggregation.required` | `shared/src/contracts/training_contracts.py` | `shared/tests/unit/test_new_training_contracts.py`의 compatibility test | 구버전 task payload를 canonical nested config로 승격하기 위함 | 저장 payload/config/report에서 legacy key가 더 이상 발견되지 않고, producer/consumer compatibility window가 종료됐을 때 |
| `TrainingTaskPayload.secure_aggregation_required` property | `TrainingTaskPayload.secure_aggregation.required` | `shared/src/contracts/training_contracts.py` | `shared/tests/unit/test_new_training_contracts.py`의 compatibility test | 구버전 bool getter를 읽는 외부 호출부가 있을 수 있음 | 외부 호출부 grep과 migration release note가 끝나고, contract test를 canonical field 기준으로 바꿀 때 |
| `TrainingTaskPayload.fssl_method`/`fssl_context`만 있는 live FSSL task | `TrainingTaskPayload.fssl_execution` + `TrainingTaskPayload.fssl_capability_plan` + context | `shared/src/contracts/training_contracts.py`, producer: `main_server/src/services/federation/rounds/round_manager_service.py`, consumer: `agent/src/services/training_runtime/current_task/runner.py` | 기존 live task tests와 저장 round payload | live FSSL runtime translation 중 하위 호환 task를 계속 읽기 위함. 새 main_server producer는 execution/capability snapshot을 함께 기록한다. | 저장 round/task payload와 API caller가 snapshot을 모두 생산하고, agent compatibility window가 끝나면 context-only method-owned task를 거부한다. |
| `adapter_kind=lora_classifier`, `lora_classifier_update`, `lora_classifier_delta.v1` | `adapter_kind=peft_classifier`, `peft_classifier_update`, `peft_classifier_delta.v2` | v2: `shared/src/contracts/adapter_contract_families/peft_classifier.py`, narrow legacy tensor fallback: materialization/checkpoint owner files | materialization/checkpoint fallback | shared v1 parser/factory, golden fixture, report verifier/result-index/dashboard fallback은 제거됐다. 기본 FL simulation config와 v2 producer는 `peft_classifier`만 쓴다. | materialization/checkpoint fallback이 필요 없다고 확인되면 남은 tensor-key fallback도 제거 |
| `communication_cost.posthoc_byte_estimates` report field | `communication_cost.artifact_byte_estimates` | `scripts/experiments/fl_ssl/federated_simulation/io/report_verification.py`, `apps/experiment_dashboard/src/app.js` | 2026-05-26 FedMatch reduced report와 dashboard cache | 기존 run/report artifact를 `runs/` 수정 없이 읽기 위함. 새 report producer와 verifier public expectation은 `artifact_byte_estimates`/`expected_communication_estimate_schema_version`만 쓴다. | historical report ingest가 새 field로 재생성되거나 old dashboard cache가 제거되면 fallback 삭제 |
| Central PEFT report `manifest.backbone.lora` key | `manifest.backbone.peft_adapter_config` | `methods/adaptation/peft_text_encoder/training/modeling.py`, `scripts/workflows/result_index/report_loader.py` | 제거 완료 | LoRA mechanism config를 PEFT adapter config surface로 승격했고 current result-index reader는 새 key만 읽는다. | 완료 |
| Dashboard bundle `central_lora_*`, `lora_*`, `adapter_family_name` run/filter fields | `central_peft_*`, `peft_adapter_*`, `payload_adapter_kind` | `apps/experiment_dashboard/src/app.js`의 bundle load normalizer | historical/local dashboard cache | `apps/experiment_dashboard/data/**`와 과거 export cache는 실험 자료라 직접 수정하지 않고, UI reader가 load 직후 current field로 정규화한다. 새 result-index export는 `peft_adapter_*`, `payload_adapter_kind`, `peft_adapter_parameters_json`만 생산하고 mechanism별 option column을 만들지 않는다. | dashboard cache가 새 result-index export로 재생성되면 app load normalizer fallback 삭제 |
| `ChildSupportResponsePlan.forbidden_terms` | `ChildSupportResponsePlan.blocked_terms` | `agent/src/features/wellbeing/child_support/response_policy.py` | repo 내부 직접 참조 없음 | 외부 또는 이전 app/runtime 호출부가 plan 검증 용어를 읽을 수 있음 | repo/app/API 소비자에서 참조가 없고, 한 compatibility window 이후 제거 가능 |
| `ChildSupportResponseStrategy` | `ChildSupportResponsePlan` | `agent/src/features/wellbeing/child_support/response_policy.py` | repo 내부 직접 참조 없음 | 이전 type name import를 깨지 않기 위한 type alias | 외부 import grep이 비어 있고, release note 또는 migration guide에 `ChildSupportResponsePlan` 전환을 명시한 뒤 |

## Completed Legacy Removals

아래 항목은 현재 compatibility 표면이 아니다. 재도입하려면
`docs/architecture/target-method-runtime-structure.md`와 관련 contract를 먼저 다시
열어야 한다.

| 제거된 표면 | 현재 상태 |
|---|---|
| `adapter_kind=diagonal_scale`, `vector_adapter_state.v1`, `vector_adapter_delta.v1` | shared parser/factory, methods/runtime 구현, update-family config leaf를 제거했다. |
| `methods/adaptation/lora_classifier/**` direct import path | repo 내부 Python import는 `methods/adaptation/peft_text_encoder/**`와 `methods/adaptation/peft_adapters/**`로 전환했고 legacy methods package는 삭제했다. |
| `scripts/experiments/fl_ssl/federated_simulation/**`와 `scripts/runtime_adapters/federated_agent/**` 안의 `lora_classifier` 실행/report alias | active Hydra root config, simulation runtime model/payload builder, FedMatch recipe, report verifier CLI/API, result-index/dashboard reader에서 제거했다. |
| `lora_classifier_materialized_state.v1` FedMatch peer snapshot kind | active FedMatch peer snapshot producer와 helper materializer는 `peft_encoder_materialized_state.v1`만 허용한다. |
| FL SSL post-run backfill CLIs | final projection과 artifact communication estimate는 active report 생성 경로가 소유한다. `backfill_final_projections.py`와 `backfill_communication_costs.py`는 삭제했다. |
| stored-event/query-buffer `selection_diagnostics_writer.py` | active FL simulation diagnostics는 report builder와 `methods/federated_ssl/diagnostics/**` helper가 소유한다. 재도입 방지는 architecture guard로 확인한다. |
| `LOCAL_SSL_POLICIES_FROM_QUERY_SSL` | active code는 `QUERY_SSL_LOCAL_OBJECTIVE_POLICY_NAMES`와 `is_query_ssl_local_objective_policy()`를 사용한다. alias는 삭제했다. |

## Update Rule

새 compatibility alias를 추가할 때는 같은 턴에서 이 ledger에 소유자, canonical 표면,
현재 소비자, 제거 조건을 추가한다. 임시 compatibility가 실험 편의만 위한 것이라면
`shared` contract가 아니라 해당 runtime/adapter 내부에 격리한다.
