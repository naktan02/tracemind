# 2026-04-02 Prototype Rebuild Embedding Boundary

## 배경

`main_server/src/services/prototypes`의 rebuild 경로가
`agent.src.infrastructure.model_adapters.embedding.factory`를 직접 import하고 있었다.

문제는 두 가지였다.

1. server-owned canonical rebuild input이 agent-owned 구현 타입을 직접 소유했다.
2. `main_server`가 runtime embedding adapter 생성 mechanism까지 직접 알게 되었다.

## 결정 내용

1. `EmbeddingAdapterSpec`을 `shared/src/domain/value_objects/`로 이동했다.
2. `main_server`의 prototype rebuild 경로는 shared spec만 의존하도록 바꿨다.
3. `StoredReferencePrototypeRebuildService`는 concrete factory를 기본값으로 숨기지 않고,
   `adapter_factory`를 명시적으로 주입받도록 변경했다.
4. Hydra embedding config의 `_target_`도 shared spec 경로를 보도록 맞췄다.

## 이유

1. `EmbeddingAdapterSpec`은 agent 구현 세부가 아니라 여러 경계에서 공유되는
   canonical 설정 값 객체다.
2. 무엇을 만들지 나타내는 spec과, 어떻게 만들지 나타내는 factory를 분리해야
   policy와 mechanism이 섞이지 않는다.
3. `main_server -> agent` 직접 import를 제거하면 추후 agent runtime 분리나
   backend 교체 시 blast radius를 줄일 수 있다.

## 보류 사항

1. concrete embedding factory protocol 자체를 shared port로 올릴지는 아직 보류한다.
2. 현재는 scripts와 tests가 agent factory를 직접 주입한다.
3. 필요 시 이후에 별도 wiring layer 또는 adapter module로 정리할 수 있다.

## 다음 액션

1. round 관련 기능을 확장할 때 `main_server/src/services/rounds/models.py`를
   domain, payload, mapper로 분리한다.
2. simulation 안정화 이후 `scripts/experiments/federated_simulation/simulation.py`
   조합 책임을 더 얇게 만든다.
