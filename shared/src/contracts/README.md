# Shared Contracts

이 디렉터리는 agent, main_server, script가 공통으로 읽는 payload 계약의 source of truth다.

문서 우선순위는 이렇게 본다.

1. 이 폴더의 Python contract 파일
2. 이 README
3. `docs/contracts/*` 설계 메모

`docs/contracts/*`는 배경과 설계 이유를 설명하는 보조 문서이고, 실제 필드 의미와 포맷은 이 폴더 파일이 기준이다.

중요한 경계:

- `shared`는 방법론 구현, 실행 profile, 기본 조합값, runtime catalog를 소유하지 않는다.
- adapter family 이름을 가진 파일은 payload parse/serialize 계약일 때만 허용한다.
- 새 FL SSL method나 local objective를 추가하기 위해 `shared` 파일을 고치는 흐름은
  잘못된 경계로 본다.
- 새 adapter family처럼 state/update payload shape 자체가 바뀌는 경우에만
  `shared/src/contracts/adapter_contract_families/`를 확장한다.

현재 활성 계획 기준:

- 초기 seed baseline은 `central + fixed embedding + classifier`다.
- query-domain 적응 단계에서만 `PEFT text encoder + linear head`를 연다.
- 시스템/FL 트랙의 우선 baseline은 `embedding -> shared scoring state -> local interpretation`이다.
- 여기서 linear classifier head는 공통 class evidence를 만드는 shared scoring artifact 중 하나로 본다.
- `PrototypePack`과 shared adapter 계열 계약은 제거 대상이 아니라 비교 실험/확장 축으로 유지한다.
- 이 README의 계약은 현재 `시스템/FL runtime`의 source of truth다. 중앙집중형 PEFT text encoder 논문 trainer는 별도 실험 레일로 다루며, paper-track scaffold는 `docs/contracts/central_peft_text_encoder_trainer_contract.md`를 기준으로 본다.

## 주요 파일

### `model_contracts.py`

서버가 배포하는 전역 model/shared-adapter manifest를 정의한다.

- `ModelManifest`
  - `model_id`, `model_revision`, `artifact_ref`, `training_scope`를 묶은 현재
    전역 shared artifact 설명
  - `auxiliary_artifact_versions`는 prototype pack처럼 주 artifact에 부속되는
    artifact version을 중립 이름으로 기록한다. prototype pack은
    `auxiliary_artifact_versions["prototype_pack"]`로만 표현한다
  - 구형 payload의 top-level `prototype_version`은 파싱 시 위 auxiliary map으로
    승격하지만 canonical dump에는 다시 쓰지 않는다
  - main server가 revision별 manifest와 active pointer를 소유한다
  - `artifact_ref`는 server-owned opaque ref이며, 파일 경로 해석은
    main server repository 내부 compatibility로만 처리한다
  - round open 요청자는 manifest를 제출하지 않고, 서버 current manifest를
    기준으로 training task가 생성된다

### `adapter_contract_families/`

Shared adapter 상태와 update payload를 정의한다.

Family별 구현은 `adapter_contract_families/base.py`,
`adapter_contract_families/classifier_head.py`,
`adapter_contract_families/peft_classifier.py`,
`adapter_contract_families/registry.py`,
`adapter_contract_families/builtin_loader.py`, `adapter_contract_families/io.py`,
`adapter_contract_families/factories.py`로 나뉜다. Registry는 lookup helper만
소유하고 builtin family 연결은 `builtin_loader.py`의 명시 목록이 담당한다.
Family별 `adapter_kind`와 canonical/accepted update payload format은 각
`adapter_contract_families/<family>.py` payload contract 옆에 둔다. 중앙
adapter family metadata catalog는 두지 않고, registry는 payload type lookup과
parse/serialize contract helper만 소유한다.
기존 compatibility facade였던 `shared.src.contracts.adapter_contracts`는 제거됐고,
runtime과 test는 family별 direct import를 사용한다.

- `SharedAdapterStatePayload`
  - 서버가 현재 배포하는 전역 shared adapter 상태 공통 필드
- `ClassifierHeadAdapterStatePayload`
  - classifier-head v1 concrete 구현
  - `head_kind="linear"`를 사용한다. `classifier_head.v1`은 generic
    classifier head family 전체가 아니라 category별 linear head payload다
  - `label_weights`, `label_biases`는 category별 linear head 파라미터다
- `PeftClassifierAdapterStatePayload`
  - PEFT-classifier v2 concrete 구현
  - LoRA/DoRA 같은 adapter mechanism은 `peft_adapter_config.peft_adapter_name`과
    `peft_adapter_config.parameters`로 표현하고, classifier head는 별도 artifact로
    유지한다
  - `adapter_kind=peft_classifier`, `schema_version=peft_classifier_state.v2`를
    사용한다
  - v1 `lora_classifier` shared parser/factory는 제거됐다. 과거 run/report artifact
    해석이 필요하면 해당 old-reader가 자기 경계에서 legacy 문자열을 읽고 즉시
    canonical PEFT 표면으로 정규화한다
- `CurrentSharedAdapterStatePayload`
  - 서버 current `ModelManifest`와 실제 `SharedAdapterStatePayload`를 함께
    내려주는 agent sync payload
  - `artifact_ref`는 서버 내부 참조일 수 있으므로 agent는 이 payload의
    inline `state`를 로컬 캐시에 저장해 사용한다
- `SharedAdapterUpdatePayload`
  - agent가 서버로 올리는 shared adapter update 공통 필드
- `ClassifierHeadAdapterUpdatePayload`
  - classifier-head v1 concrete 구현
  - `head_kind="linear"`를 사용한다
  - `label_weight_deltas`, `label_bias_deltas`는 category별 head 변화량이다
- `PeftClassifierAdapterUpdatePayload`
  - PEFT-classifier v2 update concrete 구현
  - canonical payload format은 `peft_classifier_update`다
  - PEFT adapter delta는 `peft_adapter_delta_artifact_ref` 또는
    `peft_parameter_deltas`로 표현한다. 필드명에는 LoRA를 넣지 않으며,
    실제 mechanism은 `peft_adapter_config.peft_adapter_name`이 식별한다
  - classifier head delta와 logical `partitioned_deltas` 의미는 v1과 동일하게
    유지하되, partition 이름과 sigma/psi 같은 method 의미는 `methods/`가 소유한다

### `training_objective_contracts.py`

로컬 학습 objective와 selection policy config의 canonical shape를 정의한다.

- `TrainingObjectiveConfigPayload`
  - local update objective 선택값을 담는 contract
  - `training_backend_name`은 필수 값이며 `shared`는 기본 backend를 소유하지 않는다.
    실험 기본 조합은 Hydra profile, production fallback은 runtime/default facade가
    소유한다
- `TrainingSelectionPolicyPayload`
  - 한 라운드에서 사용할 로컬 예시 선택 제한과 selection policy별 확장값을 담는다

### `training_example_backends.py`

agent training example backend 이름 중 여러 계층이 함께 읽는 canonical 값을 정의한다.

- `WEAK_STRONG_PAIR_EXAMPLE_BACKEND`
  - source row가 weak/strong view pair를 제공해야 하는 backend 이름
  - 실제 row shape 검증은 text view owner인 `methods/adaptation/query_text_views`가 맡는다
- `PROTOTYPE_RESCORE_EXAMPLE_BACKEND`
  - prototype 기반 single-view rescore backend 이름

### `secure_aggregation_contracts.py`

학습 task와 update submission이 요구하거나 제출하는 secure aggregation/encryption
metadata의 canonical shape를 정의한다.

기존 import path 호환을 위해 `SecureAggregationConfigPayload`와
`SecureAggregationSubmissionPayload`는 `training_contracts.py`에서도 계속 import할 수 있다.

- `SecureAggregationConfigPayload`
  - task가 요구하는 secure aggregation backend, encryption scheme, key/ciphertext
    metadata를 담는다
- `SecureAggregationSubmissionPayload`
  - agent가 제출한 update payload가 어떤 secure aggregation/encryption metadata를
    따르는지 담는다

### `training_contracts.py`

FL orchestration과 로컬 학습 제어용 envelope을 정의한다.

기존 import path 호환을 위해 `TrainingObjectiveConfigPayload`와
`TrainingSelectionPolicyPayload`는 이 파일에서도 계속 import할 수 있다.

- `TrainingTaskPayload`
  - 서버가 agent에 내려주는 학습 task
  - 로컬 학습 하이퍼파라미터, threshold, selection policy 포함
  - 구버전 `secure_aggregation_required` 입력/property는
    `secure_aggregation.required`로 수렴시키는 compatibility 표면이다. 유지 이유와
    제거 조건은 `docs/contracts/legacy_contract_ledger.md`를 따른다
- `TrainingUpdateEnvelopePayload`
  - 서버가 수락/저장한 update 메타데이터 봉투
  - `payload_ref`는 서버가 저장한 update payload의 opaque ref를 가리킴
  - `payload_ref`를 파일 경로로 직접 해석하지 않고, main server repository가
    ref 해석과 legacy path fallback을 담당한다
- `TrainingUpdateSubmissionPayload`
  - agent가 서버에 제출하는 update 요청
  - `envelope`과 inline `update_payload`를 함께 보내며, 서버는 payload를
    server-owned storage에 저장한 뒤 envelope의 `payload_ref`를 덮어씀
  - `envelope`의 `model_id`, `base_model_revision`, `training_scope`,
    `example_count`, `payload_format`은 inline `update_payload`와 같은 update를
    가리켜야 한다
- `DecisionFeedbackSignalPayload`
  - pseudo-label, 사용자 피드백, 후속 결과 등 로컬 학습용 signal 단위 계약

### `wellbeing_signal_contracts.py`

가족용 확장 프로그램이 읽는 wellbeing signal 출력 계약을 정의한다.

- `WellbeingSignalSummaryPayload`
  - 아이용/부모용 현재 상태 카드 source of truth
- `WellbeingSignalTimeseriesPayload`
  - 부모용 전체 추이 그래프 source of truth
- `ParentUnlockRequestPayload`
  - 부모용 상세 화면 진입용 PIN 요청
- `ParentUnlockResponsePayload`
  - 부모용 상세 화면 접근 결과와 세션 메타데이터

### `typing_segment_contracts.py`

브라우저 확장 프로그램이나 future desktop collector가 로컬 agent로 보내는
typing segment 계약을 정의한다.

중요:

- 이 계약은 extension/collector -> agent local ingest 전용이다.
- `final_text`, `deleted_text`, 입력 surface hint는 raw text 또는 raw-text-adjacent
  값이므로 main_server API나 FL update envelope으로 전달하지 않는다.
- 수집 surface가 늘어나도 producer는 `TypingSegmentPayload` canonical shape로
  정규화해서 agent에 보낸다.

- `TypingSegmentPayload`
  - `source_type`, `surface_type`, `capture_confidence`로 수집 계층과 복원 품질을
    명시한다
  - `stats`에는 insert/delete/paste/composition count 같은 편집 event 요약만 둔다
- `TypingSegmentIngestResponsePayload`
  - agent가 segment를 inference pipeline으로 처리한 결과의 최소 응답이다

### `captured_text_contracts.py`

브라우저 확장 프로그램이나 future local collector가 로컬 agent로 보내는
canonical raw text 수집 계약을 정의한다.

중요:

- 이 계약은 collector -> agent local ingest 전용이다.
- `text`, `page_url`, `page_title`, source/surface metadata는 raw text 또는
  raw-text-adjacent 값이므로 main_server API나 FL update envelope으로 전달하지
  않는다.
- 수집 source가 늘어나도 producer는 `CapturedTextEventPayload` canonical shape로
  정규화해서 agent에 보낸다.
- `source_type`과 `surface_type`은 분리한다. `reddit:comment` 같은 조합 문자열은
  agent 내부 `QueryEvent` projection에서만 만든다.

- `CapturedTextEventPayload`
  - 검색어, Reddit, webpage, typing, manual 입력을 하나의 local-only raw event로
    표현한다
  - `collector_version`과 `metadata`는 이후 재처리와 audit를 위한 provenance다
- `CapturedTextIngestResponsePayload`
  - agent가 event를 inference pipeline으로 처리한 결과의 최소 응답이다
- `CapturedTextDebugJob*Payload`
  - family extension debug page가 agent-local view generation 상태를 조회하고
    개발용 on/off, 즉시 실행을 요청하는 payload다
  - 이 payload는 production scheduler나 main_server orchestration 계약이 아니다.
    raw text와 generated weak/strong view는 agent local boundary에 남긴다

### `family_access_contracts.py`

가족용 확장 프로그램의 app-level setup/auth 경계를 정의한다.

- `FamilySetupStatusPayload`
  - 최초 setup 완료 여부와 현재 로컬 access mode
- `FamilySetupRequestPayload`
  - child/parent PIN 최초 설정 요청
- `FamilyUnlockRequestPayload`
  - role별 잠금 해제 요청
- `FamilyUnlockResponsePayload`
  - role별 세션 메타데이터와 남은 시도 횟수

### `child_support_contracts.py`

가족용 확장 프로그램의 아이용 지원 대화 API 경계를 정의한다.

- `ChildSupportConversationRequestPayload`
  - child UI가 로컬 agent에 보내는 단일 turn 입력
  - raw message는 로컬 agent API에만 전달되며 서버 공유 artifact가 아니다
- `ChildSupportConversationResponsePayload`
  - agent가 반환하는 단일 turn 응답
  - `conversation_id`는 agent-local 대화 저장소의 conversation key다
  - `assistant_mode`는 현재 응답 생성 backend가 `local_guarded`, `local_llm`, future `llm` 중 무엇인지 나타낸다
  - `safety_level`은 child UI가 과도한 내부 점수 없이 표시할 수 있는 지원 단계다
  - `scope_status`는 마음 도움 범위 안에서 답했는지, 별도 질문을 redirect했는지 나타낸다
- `ChildSupportProactivePromptPayload`
  - 아이 화면 진입 시 agent가 먼저 말을 꺼낼지 여부
  - 실제 관측 summary가 없거나 low-data이면 `should_prompt=false`로 둔다
- `ChildSupportSuggestionPayload`
  - 후속 입력 suggestion

중요:

- 대화 원문, conversation 저장, LLM provider, prompt, safety policy,
  response plan/validation 실행 방식은 agent runtime이 소유한다.
- UI는 응답을 표시하고 다음 입력을 보낼 뿐, wellbeing이나 safety 의미를 재정의하지 않는다.

### `prototype_contracts.py`

Prototype runtime이 직접 읽는 semantic artifact 계약을 정의한다.

- `PrototypePackPayload`
  - category마다 하나 이상의 prototype을 가진다
  - single prototype도 길이 1 리스트로 정규화해서 해석한다
  - v1 classifier-first baseline에서는 bootstrap/comparison artifact로도 쓴다
- `CategoryPrototypePayload`
  - `prototype_id`, `centroid`, `sample_count`를 담는다
- `extract_category_prototypes(...)`
  - runtime scoring용 `category -> prototype vectors` 변환 helper
- `extract_category_centroids(...)`
  - single-prototype pack에서만 쓰는 legacy helper

### `prototype_build_state_contracts.py`

Prototype exact incremental merge용 build-state 계약을 정의한다.

- `PrototypeBuildStatePayload`
  - 현재 v1은 category별 `embedding_sum`, `sample_count`만 담는다
  - 따라서 exact incremental merge는 single mean-centroid builder 전용이다
  - multi-prototype builder는 build-state 없이 pack만 생성할 수 있다

## 해석 규칙

- `adapter_kind`
  - adapter family discriminator
  - 예: `peft_classifier`, `classifier_head`
  - base shared adapter payload는 특정 family를 기본값으로 추정하지 않는다
  - 구형 `vector_adapter_*` / `diagonal_scale` schema fallback은 제거됐고,
    adapter kind가 없는 payload는 거부한다
- `payload_format`
  - 동일 family 안에서도 state/update envelope 해석에 쓰는 포맷 식별자
  - `TrainingUpdateSubmissionPayload`에서는 update payload type이 허용하는
    format과 일치해야 하며, main server는 active adapter family가 허용하는
    format인지 accept 단계에서 다시 확인한다
- `training_scope`
  - 어느 수준까지 학습하는지 나타내는 범위 식별자
  - 현재 시스템 runtime에서는 주로 `adapter_only`, `head_only`
  - PEFT text encoder family는 `adapter_only` 또는 `selected_encoder_block` 해석과 함께 열 가능성이 크다
  - `full_encoder`는 upper-bound 또는 미래 확장 값으로 남아 있지만, 현재 시스템 FL 기본 경로는 아니다
- `model_revision` / `base_model_revision`
  - `model_revision`: 서버가 현재 배포 중인 revision
  - `base_model_revision`: 로컬 update가 계산된 기준 revision

추가 원칙:

- `classifier_head` family는 전역 class evidence를 제공하는 shared head로 해석한다.
  현재 v1 payload shape는 `linear` head만 표현하므로, MLP/projection head처럼
  다른 shape가 필요하면 `classifier_head.v2` 또는 별도 payload family를 연다.
- `peft_classifier` family는 기존 `classifier_head`의 옵션이 아니라, PEFT
  adapter state와 classifier head state를 함께 배포/집계하는 별도 family다.
- 로컬 개인화와 최종 판단은 이 계약 파일이 아니라 agent 로컬 runtime 계층이 소유한다.
