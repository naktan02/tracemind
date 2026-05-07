# Method-Owned Runtime Refactor Plan

이 문서는 FL SSL method framework를 `methods/` 중심으로 깊게 만들기 위한 active
리팩터링 계획이다. 목적은 코드 길이 절감이 아니라 새 method 추가 시 수정 위치가
예측 가능해지고, `agent`/`main_server` runtime이 method별 구현을 흡수하지 않게
하는 것이다.

## 목표 경계

```text
shared
  contracts, domain entity, canonical payload 해석

methods
  method identity, algorithm core, SSL hook, local objective,
  adaptation update core, aggregation math, server/round policy 의미

conf
  Hydra 실행 조합과 실행값 source of truth

agent
  private/local state, raw text 접근, local artifact materialization,
  selected method core 실행 port, payload upload

main_server
  round lifecycle, validation, repository/materialization,
  selected method/server policy 실행 port, publication

scripts
  experiment entrypoint, sweep, report/artifact orchestration thin wrapper
```

새 method를 추가하기 위해 `agent`나 `main_server`에 method 이름을 가진 파일이
추가되면 실패 신호로 본다. runtime 계층에 추가할 수 있는 것은 method가 아니라
`raw_text_local_update`, `artifact_ref_materializer`, `round_state_exchange`,
`server_policy_executor` 같은 capability다.

## 수정된 Batch 순서

### Batch 0. WIP 정리와 registry primitive 잠금

상태: 완료. methods/main_server/agent 주요 registry 일부를 primitive와
implementation-local registration으로 정리했다.

남은 원칙:

- `registry.py`는 lookup/catalog/duplicate guard만 맡긴다.
- concrete builtin 목록을 `registry.py`나 `builtin_loader.py`에 계속 누적하지 않는다.
  name-to-module convention, config-declared module path, package manifest 중 하나로
  import trigger를 작게 유지한다.
- shared canonical payload family만 explicit contract source-of-truth 예외다.

### Batch 1. Local Update Runtime Seam 교정

상태: 1차 완료. Registry primitive, implementation-local catalog/registration,
selection hook 중복 제거, method-specific runtime guard를 닫았다. 단,
`builtin_loader.py` 제거는 Batch 1.5에서 별도 진행한다.

- acceptance policy는 SSL selection hook과 중복 판단을 하지 않고 metadata/compatibility
  역할로 축소한다.
- evidence/input/scoring registry는 implementation-local registration으로 정리한다.
- `shared/src/config/local_training_registry_catalog.py`는 implementation source of
  truth가 아니라 workspace/catalog compatibility snapshot으로 낮춘다.
- `agent` local update backend는 method identity를 소유하지 않는 runtime adapter로
  명명/문서화한다.
- architecture guard로 `agent`/`main_server` method-specific runtime module 추가를 막는다.

완료 기준:

- 새 FL SSL method 추가를 위해 `agent`에 method-specific backend 파일을 추가하지 않는
  규칙이 문서와 테스트에 남는다.
- 기존 local training/scoring/selection 테스트가 통과한다.

### Batch 1.5. Builtin Loader 제거

상태: 진행 중. `agent` local runtime registries는 `builtin_loader.py`를 제거하고
package-local convention import helper와 `RuntimeRegistry` primitive로 전환했다.

`builtin_loader.py`는 registry 하단 등록 block을 제거하기 위한 중간 발판이다. 최종
구조에서는 concrete module 목록을 loader에 계속 누적하지 않는다.

- 완료: `agent` acceptance/evidence/input/scoring/training registry의 import trigger를
  convention 기반 package-local import로 바꿨다.
- 완료: `agent` registry 파일의 반복 dict/register/list/catalog 코드를
  `agent/src/services/runtime_registry.py`로 공통화했다.
- 완료: `main_server` aggregation registry의 import trigger를 adapter_kind module
  convention으로 바꿨다.
- 완료: `methods/federated/aggregation` import trigger를
  adapter_kind/method_name convention과 bounded package import로 바꿨다.
- 완료: `methods/ssl/hooks` import trigger를 bounded package import로 바꿨다.
- shared canonical contract family의 explicit loader만 예외로 남긴다.

완료 기준:

- 일반 runtime/strategy registry에는 concrete 목록을 가진 `builtin_loader.py`가 남지
  않는다.
- 새 implementation 추가 시 registry나 loader 목록을 수정하지 않는다.

### Batch 2. Agent Local Update Executor Port

상태: 진행 중. `LocalTrainingService`에서 update 생성, privacy protection,
payload 저장, submission envelope 생성을 분리해 `LocalUpdateExecutor` port로 내렸다.
서비스는 model revision 검증, runtime compatibility 검증, pseudo-label selection,
accepted example 조립까지만 orchestration한다.

Batch 1에서 정리한 registry를 바탕으로 agent runtime seam을 더 깊게 만든다.

- 완료: `LocalTrainingService`가 concrete training backend default를 직접 import하지
  않고 `LocalUpdateExecutor`를 통해 selected backend/privacy guard를 resolve한다.
- 완료: architecture guard로 `LocalTrainingService`가 concrete training backend나
  training backend registry를 직접 import하지 못하게 했다.
- 완료: LoRA-classifier training row, label schema resolve, train executor contract,
  delta payload 통계 조립을 `methods/adaptation/lora_classifier/local_update.py`로
  내렸다. `agent`에는 accepted example에서 raw text를 꺼내는 adapter와 agent-local
  artifact ref glue만 남겼다.
- `methods/adaptation/<family>/`가 local update 계산 core와 method/family metadata를
  소유한다.
- `agent`는 raw text, private storage, artifact ref, payload upload를 methods core에
  연결하는 generic executor port를 둔다.
- diagonal-scale heuristic과 LoRA classifier는 각각 methods-owned core를 통해 실행되고,
  agent 파일은 capability adapter로 얇아진다.
- LoRA/backbone/tokenizer spec 중 공통 의미는 methods/shared contract 쪽 typed spec으로
  모으고, runtime-specific artifact prefix나 bootstrap ref는 agent/scripts에 남긴다.

완료 기준:

- 새 local objective나 adaptation family를 추가할 때 agent 변경이 capability 확장인지
  method-specific 누수인지 구분된다.
- 기존 payload/update manifest shape는 유지된다.

### Batch 3. Profile Source Of Truth와 Compatibility

- Hydra profile을 실행값 source of truth로 유지한다.
- Python profile/mapping은 resolver/validator 역할로 축소하거나 compatibility facade로
  낮춘다.
- LocalUpdateProfile, method descriptor, adapter family metadata, round runtime profile을
  함께 검증한다.
- incompatible 조합은 실행 중간이 아니라 bootstrap/resolve 시점에 실패한다.

완료 기준:

- FedMatch/FedLGMatch류 조합을 추가할 때 runtime core 수정이 아니라 profile/descriptor
  조합 검증으로 실패/성공이 결정된다.

### Batch 4. Main Server Round/Policy Seam

`main_server`에도 agent와 같은 문제가 생길 수 있으므로 별도 batch로 다룬다.

- `main_server/src/services/federation/rounds/families/registry.py`를 primitive +
  implementation-local registration으로 정리한다.
- aggregation 파일은 adapter family payload/state materialization만 맡고, aggregation
  math는 `methods/federated/aggregation/`에 둔다.
- method-specific server/round policy 의미는
  `methods/federated_ssl/<method>/server_policy.py`,
  `round_policy.py`에 둔다.
- `main_server`에는 generic `server_policy_executor` 또는 `round_state_exchange`
  capability만 추가한다.

완료 기준:

- FedMatch/FedLGMatch server-side 차이가 `main_server` core 파일명이나 registry branch로
  새지 않는다.

### Batch 5. SSL Hook Bundle과 Training Template

- USB식 장점인 hook 교체 구조를 TraceMind 방식으로 명시화한다.
- `SslObjectiveHooks` 같은 role-based bundle을 둬 masking, pseudo-labeling,
  consistency loss를 교체한다.
- FixMatch는 fixed-threshold hook bundle로 표현하고, FreeMatch-like test-only hook으로
  교체 가능성을 검증한다.
- fixed classifier와 LoRA classifier의 evaluation/checkpoint/history/training loop 중
  반복되는 template만 공통화한다.

완료 기준:

- 새 SSL algorithm 추가가 scripts runner 수정이 아니라 methods algorithm/hook 추가로
  보인다.
- training loop 공통화가 model input 차이를 숨기지 않고 adapter로 표현한다.

### Batch 6. Runtime Adapter와 Artifact Writer 정리

- `scripts/runtime_adapters/*`는 request mapper, repository wiring, runtime bridge로
  나눈다.
- artifact/report builder와 writer/exporter를 분리한다.
- compatibility facade는 제거 기준과 public boundary 여부를 문서화한다.
- diagnostics dataclass의 manual `to_mapping()` 반복은 serializer helper 또는 typed
  model로 줄인다.

완료 기준:

- runner는 orchestration 중심으로 남고 schema 의미나 algorithm 의미를 만들지 않는다.

### Batch 7. Extension Dry Run

- production registry에 장난감 method를 남기지 않는 방식으로 dummy FL SSL method
  extension test를 유지한다.
- 새 method 추가 시 필요한 변경이 method-local module, descriptor/config/profile,
  필요한 capability adapter뿐인지 확인한다.
- shared contract golden fixture와 architecture guard를 함께 유지한다.

완료 기준:

- 두 번째/세 번째 method 추가 비용이 테스트로 관측된다.

## Main Server 현재 점검

현재 `main_server`는 일부 정리가 되었지만 같은 위험이 남아 있다.

- 좋은 상태: aggregation registry는 primitive와 implementation-local registration으로
  낮아졌다.
- 좋은 상태: adapter family별 aggregation adapter는 methods aggregation core를 호출하는
  구조다.
- 위험: `rounds/families/registry.py`가 concrete family class와 builder function을 직접
  소유한다.
- 위험: adapter family file이 더 커지면 payload materialization과 method-specific server
  policy가 섞일 수 있다.
- 위험: experiment workspace catalog는 shared static catalog snapshot에 기대고 있어,
  runtime implementation metadata와 UI/catalog snapshot의 source-of-truth가 섞일 수 있다.
- 위험: compiler policy는 policy 판단과 config patch 조립이 한 파일에 남아 있다.

따라서 `main_server`의 다음 리팩터링 목표는 registry 정리 자체가 아니라 method-specific
server 의미가 round lifecycle로 새지 않게 하는 것이다.

## 금지 규칙

- 새 method 때문에 `agent/src/**/<method>*.py`를 만들지 않는다.
- 새 method 때문에 `main_server/src/**/<method>*.py`를 만들지 않는다.
- `shared/src/config/*catalog*.py`를 implementation registration source of truth로 키우지
  않는다.
- runtime adapter가 threshold, server weighting, pseudo-label objective 같은 method
  policy를 판단하지 않는다.
- scripts runner가 algorithm-specific registry를 새로 소유하지 않는다.

예외는 새 runtime capability가 생긴 경우뿐이다. 예외도 method 이름이 아니라 capability
이름으로 표현하고, method 의미는 `methods/`에 남긴다.
