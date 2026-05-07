---
name: tracemind-strategy-addition
description: Use when adding or replacing an adapter family, aggregation backend, training backend, scorer backend, prototype builder, or another strategy axis in TraceMind.
---

# TraceMind Strategy Addition

이 스킬은 TraceMind의 교체 가능한 전략 축을 실제 구현으로 추가할 때 쓴다.

## 언제 쓸지

- 새 `adapter_family` 추가
- 새 `aggregation_backend` 추가
- 새 scorer/example generation/training backend 추가
- prototype builder 계열 확장

## 작업 순서

1. 어떤 변화 축인지 명확히 적는다.
   - adapter family
   - aggregation backend
   - scorer backend
   - training backend
   - prototype builder
2. 아래 순서로 읽는다.
   - root `AGENTS.md`
   - `docs/project_execution_plan.md`
   - `docs/strategy_surface_map.md`
   - `docs/contracts/strategy_addition_playbook.md`
   - 관련 path-specific `AGENTS.md`
   - 새 Protocol이나 전략 축 구현 세부가 필요할 때만
     `docs/contracts/algorithm_extension_guide.md`
3. 소유 경계를 먼저 정한다.
   - canonical rule은 `shared`
   - local runtime은 `agent`
   - orchestration은 `main_server`
   - experiment wrapper는 `scripts`
4. family/strategy interface와 registration을 함께 맞춘다.
5. config source of truth를 한 군데에만 둔다.
6. 관련 테스트와 문서를 같이 닫는다.

## 설계 스타일

전략 추가는 raw registry entry를 늘리는 작업으로 끝내지 않는다. 바뀌는 축에 따라
아래 패턴을 조합한다.

- method identity와 capability가 중요하면 descriptor를 둔다.
- tensor objective의 일부만 바뀌면 hook 또는 hook bundle을 둔다.
- local/server/runtime 조합이 바뀌면 typed profile과 compatibility validator를 둔다.
- runtime framework 차이는 adapter에 두고, strategy 의미는 `methods/` 또는 계약
  가까이에 둔다.
- 단일 implementation 전용 helper는 method-local module에 두고, 두 개 이상 전략에서
  의미가 안정되면 공통 module로 승격한다.

## 체크리스트

- 전략 축이 다른 이유로 바뀌는 로직과 섞이지 않는가
- raw registry를 핵심 도메인 추상화로 남용하지 않는가
- producer/consumer가 같은 discriminator와 payload를 보는가
- 새 implementation 추가가 기존 구현 파괴 없이 확장으로 보이는가
