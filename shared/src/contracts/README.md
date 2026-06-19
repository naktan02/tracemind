# Shared Contracts

이 디렉터리는 agent, main_server, scripts가 공통으로 읽는 payload 계약의 source of
truth다. 실제 필드 의미와 포맷은 Python contract 파일이 기준이고, 이 README는 어떤
파일을 언제 읽어야 하는지 안내한다.

문서 우선순위:

1. `shared/src/contracts/*.py`
2. `shared/src/contracts/adapter_contract_families/*.py`
3. 이 README
4. `docs/contracts/*` 설계 보조 문서

`docs/contracts/*`는 배경과 설계 이유를 설명한다. 필드 의미를 바꿀 때는 계약 파일과
producer/consumer/test를 함께 바꾼다.

## Boundaries

`shared`가 소유하는 것:

- agent, main_server, scripts가 함께 해석해야 하는 payload shape
- model manifest, training task, training update envelope
- shared adapter state/update family
- scoring/runtime profile/secure aggregation 같은 cross-runtime metadata
- canonical parse/serialize와 compatibility rule

`shared`가 소유하지 않는 것:

- 방법론 구현과 SSL/FSSL objective
- Hydra 실행 preset과 runtime 기본 조합
- agent-local captured text, child support, family access, wellbeing UI payload
- FastAPI route, repository, artifact storage 구현
- dashboard view-model과 report cache

Agent-local API/UI 계약은 `agent/src/contracts/`가 소유한다.

## Contract Map

| File | Purpose |
|---|---|
| `model_contracts.py` | server-published `ModelManifest`와 active shared artifact 설명 |
| `training_contracts.py` | `TrainingTaskPayload`, update envelope/submission, decision feedback signal |
| `training_objective_contracts.py` | local objective와 selection policy config shape |
| `training_example_backends.py` | 여러 계층이 함께 읽는 training example backend 이름 |
| `scoring_contracts.py` | inference/validation scoring backend와 score policy config |
| `agent_runtime_profile_contracts.py` | server가 agent에 내려주는 runtime profile과 validation payload |
| `fl_round_contracts.py` | FL round API/runtime에서 공유하는 round payload |
| `secure_aggregation_contracts.py` | secure aggregation/encryption config와 submission metadata |
| `labeled_query_row_contracts.py` | labeled query row artifact payload |
| `registry_catalog_metadata.py` | registry/catalog display metadata shape |
| `common_types.py` | 여러 contract가 함께 쓰는 small common types |

## Adapter Contract Families

`adapter_contract_families/`는 shared adapter state와 update payload를 family별로
정의한다.

| File | Purpose |
|---|---|
| `base.py` | family 공통 state/update protocol |
| `classifier_head.py` | category별 linear classifier head state/update |
| `peft_classifier.py` | PEFT adapter와 classifier head를 함께 다루는 state/update |
| `registry.py` | payload family lookup primitive |
| `builtin_loader.py` | builtin family 명시 로딩 |
| `io.py` | parse/serialize helper |
| `factories.py` | contract payload factory helper |

Registry는 lookup helper만 소유한다. concrete implementation import와 중앙 등록 목록은
`builtin_loader.py`의 명시 목록이 맡는다.

## Core Meanings

- `artifact_ref`
  - server-owned opaque ref다. 파일 경로로 직접 해석하지 않고 main_server repository가
    ref 해석과 legacy fallback을 담당한다.
- `adapter_kind`
  - shared adapter family discriminator다. 예: `classifier_head`, `peft_classifier`.
  - base payload는 특정 family를 기본값으로 추정하지 않는다.
- `payload_format`
  - 동일 family 안에서 state/update envelope 해석에 쓰는 format 식별자다.
  - update submission에서는 inline update payload type이 허용하는 format과 일치해야 한다.
- `training_scope`
  - 어느 수준까지 학습하는지 나타내는 범위 식별자다.
  - 현재 runtime에서는 주로 `head_only`, `adapter_only`, `adapter_and_head`를 사용한다.
- `model_revision` / `base_model_revision`
  - `model_revision`은 서버가 배포 중인 revision이다.
  - `base_model_revision`은 로컬 update가 계산된 기준 revision이다.

## Runtime Interpretation

- `classifier_head` family는 전역 class evidence를 제공하는 shared linear head로 해석한다.
- `peft_classifier` family는 PEFT adapter state와 classifier head state를 함께
  배포/집계하는 family다.
- 로컬 개인화와 최종 판단은 shared contract가 아니라 agent local runtime이 소유한다.
- aggregation arithmetic은 `methods/federated/aggregation/`이 소유한다.
- adapter family delta 해석과 next-state projection은 `methods/adaptation/<family>/`가
  소유한다.
- server round lifecycle과 artifact publication은 `main_server`가 소유한다.

## Compatibility

legacy format이나 임시 변환이 필요하면 핵심 contract path에 섞지 않고 compatibility
계층으로 격리한다. 현재 남아 있는 legacy 표면과 제거 조건은
[../../../docs/contracts/legacy_contract_ledger.md](../../../docs/contracts/legacy_contract_ledger.md)를
기준으로 본다.
