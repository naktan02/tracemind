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
| `methods/adaptation/lora_classifier/**` direct import path | `methods/adaptation/text_classifier/peft_encoder/**`, `methods/adaptation/text_classifier/aggregation/**`, `methods/adaptation/peft_adapters/**` | `methods/adaptation/lora_classifier/README.md`, `tests/architecture/test_layer_dependencies.py` | repo 내부 Python import는 새 경로로 전환 완료. legacy path 자체는 기존 artifact, 외부 notebook/script, 구버전 runtime import 호환 표면으로만 남김 | shared contract와 report/runtime naming이 아직 `lora_classifier` adapter family를 canonical value로 사용함 | `text_classifier_peft_*` 계열 shared contract v2와 artifact/report migration이 끝나고, 기존 run artifact/verifier compatibility window가 종료됐을 때 |
| `adapter_kind=lora_classifier`, `lora_classifier_update`, `lora_classifier_delta.v1` | 후보: `text_classifier_peft_encoder`, `text_classifier_peft_update`, `text_classifier_peft_delta.v2` | `shared/src/contracts/adapter_contract_families/lora_classifier.py` | shared contract, FL simulation/runtime configs, report verifier, 기존 run artifact | adapter family 식별자와 payload format은 cross-boundary contract라 폴더 이동과 동시에 바꾸면 producer/consumer drift가 큼 | producer, consumer, verifier, dashboard/report fixture가 v2 이름을 동시에 지원하고 v1 legacy ledger가 별도 compatibility path로 격리됐을 때 |
| `ChildSupportResponsePlan.forbidden_terms` | `ChildSupportResponsePlan.blocked_terms` | `agent/src/services/wellbeing/child_support_response_policy.py` | repo 내부 직접 참조 없음 | 외부 또는 이전 app/runtime 호출부가 plan 검증 용어를 읽을 수 있음 | repo/app/API 소비자에서 참조가 없고, 한 compatibility window 이후 제거 가능 |
| `ChildSupportResponseStrategy` | `ChildSupportResponsePlan` | `agent/src/services/wellbeing/child_support_response_policy.py` | repo 내부 직접 참조 없음 | 이전 type name import를 깨지 않기 위한 type alias | 외부 import grep이 비어 있고, release note 또는 migration guide에 `ChildSupportResponsePlan` 전환을 명시한 뒤 |

## Update Rule

새 compatibility alias를 추가할 때는 같은 턴에서 이 ledger에 소유자, canonical 표면,
현재 소비자, 제거 조건을 추가한다. 임시 compatibility가 실험 편의만 위한 것이라면
`shared` contract가 아니라 해당 runtime/adapter 내부에 격리한다.
