---
name: architecture-first-modular-engineering
description: Use when a project needs a reusable engineering style that prioritizes contract-first design, clear change-axis separation, source-adjacent documentation, and explicit global/shared vs local/private boundaries. Apply for refactors, architecture reviews, shared interface design, reusable repo instructions, or when deciding between strategy/factory/registry patterns.
---

# Architecture-First Modular Engineering

이 스킬은 "클래스를 많이 나누는 것"보다 "변화 축을 분리하고, 계약을 source of truth로 두는 것"을 우선하는 작업 스타일이다.

## 언제 쓸지

- 장기 유지보수 가능한 리팩터링
- 계약/도메인 경계 설계
- global/shared 와 local/private 책임 분리
- heuristic/gradient, runtime/trainer/aggregator 같이 바뀔 축 분리
- repo-wide 지침을 재사용 가능한 형태로 만들기

## 기본 절차

1. 먼저 바뀔 축을 적는다.
   - 예: adapter family, local training algorithm, server aggregation, privacy layer
2. 그 축마다 공통 계약과 구현체를 분리한다.
3. source of truth는 구현 가까이에 둔다.
   - 계약 필드 의미는 계약 파일 옆에 둔다.
   - 별도 설계 문서는 배경 설명으로만 둔다.
4. raw registry를 기본 해법으로 두지 않는다.
   - 단순 문자열 매핑이면 registry도 가능하다.
   - payload 해석, 허용 포맷, runtime/backend 조합처럼 "행동 묶음"이면 strategy/factory/family object를 우선한다.
5. global/shared 와 local/private 를 구분한다.
   - 공통 의미 표현, 안전한 집계 대상, 충분한 공통 데이터가 있는 것은 global/shared
   - 개인 해석, privacy 민감 값, 평균내면 왜곡되는 것은 local/private
6. repo instruction은 계층적으로 둔다.
   - repo-wide 공통 지침 1장
   - 필요한 경로에만 path-specific 지침
   - 개인 취향은 repo 밖 전역 레이어

## 판단 규칙

### 무엇을 global로 둘지

- 사용자 간 의미가 비교적 안정적이다
- 여러 client를 합쳐도 왜곡이 작다
- 공유해도 privacy/harm radius가 상대적으로 낮다

### 무엇을 local로 둘지

- 사람마다 해석 차이가 크다
- 초기 편향이 전역으로 퍼지면 위험하다
- 개인 threshold, personal prototype, persistence 같은 값이다

## 패턴 선택 규칙

- `Strategy`
  - 같은 계약 위에 알고리즘만 바뀔 때
- `Abstract Factory` 또는 family object
  - 하나의 선택이 runtime/trainer/aggregator/codec를 같이 바꿀 때
- `Registry`
  - composition root에서 구현체를 찾는 얇은 wiring 용도일 때만
- `Port/Adapter`
  - privacy, storage, transport처럼 외부 의존을 바꿔야 할 때

## 문서 규칙

- 필드 의미는 계약 파일에 직접 드러나야 한다.
- 별도 문서는 설계 이유, 배경, 예외 규칙만 둔다.
- "실제로 뭘 의미하는가"를 찾기 위해 docs부터 뒤지게 만들지 않는다.

## 결과물 체크리스트

- 바뀔 축이 코드에서 독립적으로 보이는가
- global/local/privacy 경계가 섞이지 않는가
- 계약 파일만 읽어도 필드 의미가 이해되는가
- repo-wide 지침과 path-specific 지침이 충돌하지 않는가
- 새 implementation 추가 시 기존 구현을 뜯지 않고 확장 가능한가

필요하면 [instruction_layers.md](references/instruction_layers.md)를 읽고,
도구별 적용 형식을 고른다.
