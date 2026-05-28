# LoRA Classifier V1 Terminology Audit

이 문서는 `lora_classifier` 용어가 남아 있는 표면을 분류한다. 현재 source of truth는
코드와 code-adjacent README이며, 이 문서는 v2 migration이 끝날 때까지 바꿔도 되는
것과 바꾸면 안 되는 것을 구분하는 보조 기준이다.

## 결론

`lora_classifier`는 더 이상 구현 폴더의 source-of-truth 이름이 아니다. 구현 core는
`methods/adaptation/peft_text_classifier/`,
`methods/adaptation/peft_text_classifier/aggregation/`,
`methods/adaptation/peft_adapters/`가 소유한다.

`lora_classifier` shared parser/factory는 제거됐다. 이제 이 이름은 active
producer나 shared contract가 아니라 old artifact/report/materialization reader가
과거 산출물을 해석할 때만 읽는 legacy discriminator다.

신규 v2 shared contract 이름은 `peft_classifier`다. `adapter_kind=peft_classifier`,
`payload_format=peft_classifier_update`, `schema_version=peft_classifier_state.v2` /
`peft_classifier_delta.v2`를 사용한다. v2는 LoRA/DoRA 같은 mechanism을
`peft_adapter_config.peft_adapter_name`으로 표현하고, classifier head는 별도 구성요소로
유지한다.

## 유지해야 하는 v1 표면

| 표면 | 예 | 유지 이유 | 변경 조건 |
|---|---|---|---|
| shared schema/version | `lora_classifier_state.v1`, `lora_classifier_delta.v1` | shared parser/factory와 golden fixture에서는 제거됐다. 과거 artifact가 남아 있으면 old-reader가 자기 경계에서만 처리한다. | old artifact reader compatibility window가 끝날 때 |
| adapter kind | `adapter_kind=lora_classifier` | active config는 `peft_classifier`를 쓴다. 과거 report/result loader만 legacy 값을 canonical payload adapter kind로 정규화한다. | old artifact reader compatibility window가 끝날 때 |
| payload format | `lora_classifier_update` | active training update envelope에서는 제거됐다. 과거 envelope를 읽어야 하는 old-reader fallback에만 남긴다. | old-run reader compatibility window가 끝날 때 |
| Hydra runtime config | `round_runtime.lora_classifier` | active root config에서는 제거했다. 기존 config/report 해석이 필요하면 old-run reader compatibility에서만 다룬다. | old-run reader compatibility window가 끝날 때 |
| report/artifact path | `lora_classifier` aggregate snapshot path, report expectation field | 기존 run artifact와 report verification이 읽는다. | artifact reader가 v1/v2를 모두 읽고 compatibility window가 끝날 때 |
| registry names | `lora_classifier_trainer`, `lora_classifier_eval` | active local-update profile leaf에서는 제거했다. old-run/result reader와 v1 payload compatibility에서만 읽는다. | old-run reader compatibility window가 끝날 때 |

## 구현 Source Of Truth

| 구현 의미 | 현재 source of truth |
|---|---|
| PEFT text classifier config/training/evaluation | `methods/adaptation/peft_text_classifier/` |
| PEFT encoder aggregation/state projection | `methods/adaptation/peft_text_classifier/aggregation/` |
| LoRA/RSLoRA adapter mechanism | `methods/adaptation/peft_adapters/` |
| FedMatch method semantics | `methods/federated_ssl/fedmatch/` |
| Query row/view/token-batch glue | `methods/adaptation/query_text_views/` |

삭제된 `methods/adaptation/lora_classifier/**` direct import path는 repo 내부
compatibility surface가 아니다. 구현 source of truth는
`methods/adaptation/peft_text_classifier/**`이고, `lora_classifier` 이름은 artifact
reader, materialization fallback, legacy report field compatibility에만 남긴다.
새 business logic, source-of-truth constant, method/runtime policy를 추가하지 않는다.

## 바꿔도 되는 표면

- active docs에서 구현 owner를 설명하는 경로는 새 source-of-truth 경로로 갱신한다.
- 단위 테스트 import는 legacy path 대신 canonical path를 사용한다.
- architecture guard는 legacy path에 새 구현 파일이 생기지 않게 막는다.

## v2 Rename 작업 단위

v2 rename은 단일 파일 rename이 아니라 cross-boundary migration이다. 최소 작업 단위는
아래를 함께 포함해야 한다.

- shared contract v2 schema/version과 old-run reader 정규화
- producer: training update envelope, local update payload builder, artifact writer
- consumer: server materialization, aggregation, validation/evaluation
- config: Hydra runtime/profile alias와 migration guide
- verifier/report: report verifier, fixture, result index reader
- docs: `legacy_contract_ledger.md`, shared contract docs, runtime docs

이 범위를 한 번에 닫지 못하면 v1 이름을 유지한다.
