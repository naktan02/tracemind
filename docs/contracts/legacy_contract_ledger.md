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
| `adapter_kind=diagonal_scale`, `vector_adapter_state.v1`, `vector_adapter_delta.v1` | 제거됨. 신규 canonical 표면 없음 | `shared/src/contracts/adapter_contract_families/*`, `tests/architecture/test_layer_dependencies.py` | 없음. shared parser/factory와 methods/runtime 구현 모두 제거 | target update-family 축이 아니고 현재 실행/old-reader 요구가 없어 compatibility를 유지하지 않는다 | 완료. 재도입하려면 새 target structure 문서와 shared v2 계약부터 다시 열어야 한다. |
| `methods/adaptation/lora_classifier/**` direct import path | `methods/adaptation/peft_text_classifier/**`, `methods/adaptation/peft_adapters/**` | `tests/architecture/test_layer_dependencies.py`, `docs/contracts/lora_classifier_v1_terminology_audit.md` | repo 내부 Python import는 새 경로로 전환 완료했고 legacy methods package는 삭제했다. v1 shared contract 이름은 artifact/reader compatibility 표면으로만 남김 | 기존 외부 notebook/script가 삭제된 Python import path를 직접 참조할 수 있음 | 완료. 외부 direct import compatibility는 더 이상 repo 내부 package로 보장하지 않는다. |
| `adapter_kind=lora_classifier`, `lora_classifier_update`, `lora_classifier_delta.v1` | `adapter_kind=peft_classifier`, `peft_classifier_update`, `peft_classifier_delta.v2` | v2: `shared/src/contracts/adapter_contract_families/peft_classifier.py`, narrow legacy tensor fallback: materialization/checkpoint owner files | materialization/checkpoint fallback | shared v1 parser/factory, golden fixture, report verifier/result-index/dashboard fallback은 제거됐다. 기본 FL simulation config와 v2 producer는 `peft_classifier`만 쓴다. | materialization/checkpoint fallback이 필요 없다고 확인되면 남은 tensor-key fallback도 제거 |
| `scripts/experiments/fl_ssl/federated_simulation/**`와 `scripts/runtime_adapters/federated_agent/**` 안의 `lora_classifier` key, verifier flag, legacy evaluator name, bridge alias | `round_runtime.peft_classifier`, `peft_classifier_eval`, `simulation_peft_classifier_state_ref`, `expect_peft_classifier_aggregate_snapshot`, canonical implementation owner는 `methods/adaptation/peft_text_classifier/**` | `io/report_verification*.py`, `scripts/experiments/result_index/**`, `methods/adaptation/peft_text_classifier/aggregation/peft_encoder_*_projection.py`, `tests/architecture/test_layer_dependencies.py` | 없음 | active Hydra root config, simulation runtime model/payload builder, FedMatch recipe, report verifier CLI/API, result-index/dashboard reader에서 legacy `lora_classifier` 실행/report alias를 제거했다. | 완료 |
| `lora_classifier_materialized_state.v1` FedMatch peer snapshot kind | `peft_encoder_materialized_state.v1` | `methods/adaptation/peft_text_classifier/federated_ssl/peer_predictions.py` | FedMatch helper provider, resume/checkpoint compatibility | active FedMatch peer snapshot producer는 PEFT encoder kind를 쓰고, helper materializer는 기존 checkpoint의 v1 kind를 읽기 위해 fallback만 허용한다. active unit fixture가 v1 kind를 직접 만들지 않도록 architecture guard로 고정한다. | 기존 checkpoint/resume artifact compatibility window가 끝나고 `PEFT_ENCODER_ACCEPTED_PEER_SNAPSHOT_KINDS`에서 legacy kind를 제거할 때 |
| `ChildSupportResponsePlan.forbidden_terms` | `ChildSupportResponsePlan.blocked_terms` | `agent/src/services/wellbeing/child_support_response_policy.py` | repo 내부 직접 참조 없음 | 외부 또는 이전 app/runtime 호출부가 plan 검증 용어를 읽을 수 있음 | repo/app/API 소비자에서 참조가 없고, 한 compatibility window 이후 제거 가능 |
| `ChildSupportResponseStrategy` | `ChildSupportResponsePlan` | `agent/src/services/wellbeing/child_support_response_policy.py` | repo 내부 직접 참조 없음 | 이전 type name import를 깨지 않기 위한 type alias | 외부 import grep이 비어 있고, release note 또는 migration guide에 `ChildSupportResponsePlan` 전환을 명시한 뒤 |

## Update Rule

새 compatibility alias를 추가할 때는 같은 턴에서 이 ledger에 소유자, canonical 표면,
현재 소비자, 제거 조건을 추가한다. 임시 compatibility가 실험 편의만 위한 것이라면
`shared` contract가 아니라 해당 runtime/adapter 내부에 격리한다.
