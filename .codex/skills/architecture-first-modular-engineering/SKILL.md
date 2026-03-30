---
name: architecture-first-modular-engineering
description: Use when a project needs a reusable engineering style that prioritizes contract-first design, clear change-axis separation, source-adjacent documentation, and explicit separation between shared/common concerns and context-specific concerns. Apply for refactors, architecture reviews, shared interface design, reusable repo instructions, or when choosing patterns based on variation structure.
---

# Architecture-First Modular Engineering

이 스킬은 "패턴 이름을 고르는 것"보다 "문제의 변화 구조를 먼저 읽고, 그에 맞는 경계를 세우는 것"을 우선하는 작업 스타일이다.

## 언제 쓸지

- 장기 유지보수 가능한 리팩터링
- 계약/도메인 경계 설계
- 공통 계층과 문맥별 계층 분리
- 함께 바뀌는 것과 독립적으로 바뀌어야 하는 것 분리
- repo-wide 지침을 재사용 가능한 형태로 만들기

## 기본 절차

1. 먼저 문제 구조를 적는다.
   - 무엇이 자주 바뀌는지
   - 무엇이 안정적으로 유지돼야 하는지
   - 무엇이 함께 바뀌는지
   - 무엇이 독립적으로 바뀌어야 하는지
2. 그 구조에 맞춰 계약, 상태 소유권, 책임 경계를 세운다.
3. source of truth는 구현 가까이에 둔다.
   - 계약 필드 의미는 계약 파일 옆에 둔다.
   - 별도 설계 문서는 배경 설명으로만 둔다.
4. 패턴은 문제에 맞춰 고른다.
   - 알고리즘 교체면 `Strategy`
   - 생성 조합 교체면 `Factory`
   - 상태 단계가 중요하면 `State`
   - 규칙 판단이 핵심이면 `Policy` 또는 `Specification`
   - 처리 흐름이 핵심이면 `Pipeline`
   - 외부 의존 분리가 핵심이면 `Port/Adapter`
   - 횡단 기능이면 `Decorator`
   - 단순 구현체 조회면 `Registry`
5. 공통 계층과 문맥별 계층을 구분한다.
   - 여러 문맥에서 안정적으로 공유되는 것은 공통 계층
   - 사용자, 환경, 기능, 조직에 따라 달라지는 것은 문맥별 계층
6. repo instruction은 계층적으로 둔다.
   - repo-wide 공통 지침 1장
   - 필요한 경로에만 path-specific 지침
   - 개인 취향은 repo 밖 전역 레이어

## 판단 규칙

### 무엇을 공통 계층에 둘지

- 여러 문맥에서 안정적으로 유지된다
- 합쳐도 의미 왜곡이 작다
- 잘못돼도 영향 반경이 상대적으로 작다

### 무엇을 문맥별 계층에 둘지

- 사용자, 환경, 기능별로 해석 차이가 크다
- 초기 편향이 공통 계층으로 퍼지면 위험하다
- 민감한 상태나 개인화된 해석을 가진다

## 패턴 선택 규칙

- 패턴 자체를 목적처럼 쓰지 않는다.
- 항상 "무엇이 왜 바뀌는가"를 기준으로 선택한다.
- 한 패턴으로 모든 문제를 풀려 하지 않는다.
- 필요하면 여러 패턴을 조합한다.

## 문서 규칙

- 필드 의미는 계약 파일에 직접 드러나야 한다.
- 별도 문서는 설계 이유, 배경, 예외 규칙만 둔다.
- 같은 배경 설명과 계획을 여러 긴 문서에 반복하지 않는다.

## 결과물 체크리스트

- 바뀔 축이 코드에서 독립적으로 보이는가
- 공통 계층과 문맥별 계층이 섞이지 않는가
- 계약 파일만 읽어도 필드 의미가 이해되는가
- repo-wide 지침과 path-specific 지침이 충돌하지 않는가
- 새 implementation 추가 시 기존 구현을 뜯지 않고 확장 가능한가

필요하면 [instruction_layers.md](references/instruction_layers.md)를 읽고,
도구별 적용 형식을 고른다.
