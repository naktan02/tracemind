# Methods

`methods/`는 TraceMind 실험과 runtime adapter가 함께 재사용하는 알고리즘 core
패키지다. SSL objective, adaptation/update family, classifier primitive, FL aggregation,
FL SSL method descriptor와 policy 의미를 소유한다.

`methods`는 실행 환경을 소유하지 않는다. Hydra entrypoint, FastAPI route, repository,
artifact storage, raw text/private state, report/dashboard 생성은 각각 `conf/`,
`scripts/`, `agent`, `main_server`, `apps`의 책임이다.

## What Belongs Here

- SSL objective, pseudo-labeling, thresholding, consistency loss
- PEFT/text encoder adaptation과 local update 계산 core
- fixed-feature/classification primitive
- privacy guard와 local objective regularizer 계산
- federated aggregation의 순수 계산 core
- FL SSL method descriptor, local objective, server/round policy 의미
- 여러 runtime이 호출할 수 있는 methods-owned request/result surface

## What Does Not Belong Here

- Hydra config loading과 CLI orchestration
- FastAPI route, HTTP transport, repository, state storage
- main_server round lifecycle, update acceptance, artifact publication
- agent-local raw text 접근, personal threshold, private state
- report writer, dashboard view-model, 논문 표/그림 생성

## Dependency Direction

```text
shared
  ↑
methods
  ↑
agent / main_server / scripts
```

`methods`는 `shared`와 외부 ML 라이브러리만 import한다. `agent`, `main_server`,
`scripts`를 import하지 않는다.

새 method core는 기본적으로 `methods/`와 `conf/`에 추가한다. `agent`나
`main_server`에 method 이름을 가진 runtime 파일이 늘어나면 runtime adapter가 method
framework 역할을 흡수하고 있다는 신호다.

## Read Path

| Goal | Start here |
|---|---|
| 중앙/FL SSL objective를 본다 | [ssl/README.md](ssl/README.md) |
| FL SSL method descriptor와 FedMatch를 본다 | [federated_ssl/README.md](federated_ssl/README.md) |
| PEFT/full text encoder adaptation을 본다 | [adaptation/README.md](adaptation/README.md) |
| fixed-feature 지도학습 baseline core를 본다 | [classification/fixed_feature/README.md](classification/fixed_feature/README.md) |
| linear classifier head primitive를 본다 | [classification/linear_head/README.md](classification/linear_head/README.md) |
| FedAvg와 generic aggregation core를 본다 | [federated/aggregation/README.md](federated/aggregation/README.md) |
| non-IID client shard 계산을 본다 | [federated/README.md](federated/README.md) |
| 평가 metric helper를 본다 | [evaluation/README.md](evaluation/README.md) |

최종 method/runtime 구조와 migration 판단은
[../docs/architecture/target-method-runtime-structure.md](../docs/architecture/target-method-runtime-structure.md)를
우선한다.

## Package Map

| Package | Responsibility |
|---|---|
| `ssl/` | FixMatch, FreeMatch, AdaMatch 등 SSL objective framework와 shared hooks/primitives |
| `adaptation/` | trainable surface/update family, PEFT adapter, text encoder training, privacy guard |
| `classification/` | fixed-feature baseline, modality-independent linear head primitive |
| `federated/` | FedAvg 같은 reusable aggregation core와 shard policy |
| `federated_ssl/` | FedMatch 같은 FL SSL method descriptor, capability plan, method-owned policy |
| `evaluation/` | 중앙 SSL과 FL SSL이 공유하는 metric 계산 helper |
| `common/` | 여러 methods package가 공유하는 small utility |

## How To Add New Logic

### New SSL objective

1. [ssl/NEW_METHOD.md](ssl/NEW_METHOD.md)를 먼저 본다.
2. `methods/ssl/algorithms/<method>/`에 method-local core를 둔다.
3. 기존 hook/primitives로 표현 가능한지 확인한다.
4. 필요한 Hydra leaf는 `conf/strategy_axes/ssl_objective/consistency_method/`에 둔다.

Algorithm-local hook은 처음에는 해당 method 아래에 둔다. 두 개 이상 algorithm에서
같은 의미로 안정적으로 쓰일 때만 `methods/ssl/hooks/` 또는 `methods/ssl/primitives/`
로 승격한다.

### New trainable/update family

1. `methods/adaptation/<family>/`가 family-specific training/update/projection을 소유한다.
2. shared payload shape가 바뀌면 `shared/src/contracts/adapter_contract_families/`를 함께 연다.
3. runtime이 고를 설정은 `conf/strategy_axes/model_architecture/update_family/`에 둔다.
4. scripts/agent/main_server는 family 이름으로 분기하지 않고 declared callable/capability를 호출한다.

### New FL SSL method

1. [federated_ssl/NEW_METHOD.md](federated_ssl/NEW_METHOD.md)를 먼저 본다.
2. `methods/federated_ssl/<method>/`에 descriptor, method surface, original spec,
   local objective, method-only policy를 둔다.
3. 여러 method가 공유할 수 있는 mechanism만 `federated_ssl/hooks/` 또는 capability axis로 올린다.
4. `agent`와 `main_server`에는 method-specific 파일을 만들지 않는다.

## Naming Notes

- `peft_text_encoder`는 text encoder, tokenizer, PEFT adapter, head/scorer가 함께
  움직이는 update family다.
- `linear_head`는 modality-independent classifier head primitive다.
- `peft_classifier`는 shared payload adapter kind다.
- `lora_classifier`, `adapter_family_name`, `diagonal_scale`은 legacy/historical 이름이다.
  새 실행 config와 report/result reader에서는 `peft_text_encoder`,
  `payload_adapter_kind`, `update_family_name`, `trainable_state` 용어를 쓴다.

구현 상태와 기본 선택값은 [../conf/README.md](../conf/README.md)와 실제
`conf/strategy_axes/**` leaf를 기준으로 본다. 이 문서는 `methods/`의 책임 경계와
읽기 경로만 설명한다.
