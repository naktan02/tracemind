---
trigger: always_on
---

---

name: architecture-first-modular-engineering
description: Use when a project needs a reusable engineering style that prioritizes contract-first design, clear change-axis separation, canonical representations, source-adjacent documentation, explicit separation between common and context-specific concerns, and pattern choice based on problem structure rather than pattern preference.

---

# Architecture-First Modular Engineering

이 스킬은 "패턴 이름을 고르는 것"보다 "문제의 변화 구조를 먼저 읽고, 그에 맞는 경계와 계약을 세우는 것"을 우선하는 작업 스타일이다.

## 언제 쓸지

- 장기 유지보수 가능한 리팩터링
- 계약/도메인 경계 설계
- 공통 계층과 문맥별 계층 분리
- 함께 바뀌는 것과 독립적으로 바뀌어야 하는 것 분리
- repo-wide 지침을 재사용 가능한 형태로 만들기

## 핵심 원칙

1. 구현보다 먼저 문제 구조를 본다.
   - 무엇이 자주 바뀌는지
   - 무엇이 안정적으로 유지돼야 하는지
   - 무엇이 함께 바뀌는지
   - 무엇이 서로 독립적으로 바뀌어야 하는지
   - 무엇이 공통 관심사이고 무엇이 문맥별/사용자별/환경별 관심사인지 먼저 적는다.
2. 계약과 경계를 먼저 세운다.
   - 도메인 객체
   - 입력/출력 계약
   - 책임 경계
   - 상태 소유권
   - 의존 방향
     을 구현보다 먼저 정의하거나 검증한다.
3. 책임은 의미 기준으로 분리한다.
   - 파일 수보다 역할과 변화 축을 기준으로 나눈다.
   - 서로 다른 이유로 바뀌는 것은 같은 곳에 두지 않는다.
4. 공통성과 특수성을 구분한다.
   - 여러 문맥에서 안정적으로 공유되는 것은 공통 계층에 둔다.
   - 문맥, 사용자, 환경, 조직, 기능별로 달라지는 것은 지역 계층으로 분리한다.
5. 패턴 이름을 고집하지 않는다.
   - 패턴은 문제 구조에 맞춰 선택한다.
   - 특정 패턴을 미리 정해 두고 억지로 끼워 맞추지 않는다.
6. 하드코딩된 강한 결합을 줄인다.
   - 하나를 바꾸기 위해 여러 계층을 동시에 뜯어야 하면 구조 문제로 본다.
7. source of truth는 구현 가까이에 둔다.
   - 중요한 필드 의미, 계약 의미, 해석 규칙은 코드 또는 코드 가까운 문서에서 바로 보여야 한다.
8. 문서는 짧고 역할이 분명해야 한다.
   - 같은 배경 설명과 계획을 여러 문서에 반복하지 않는다.
9. 미래 교체 비용을 계속 점검한다.
   - 핵심 도메인 모델, 실행 방식, 저장 방식, 외부 연동, 규칙/정책, 보안/프라이버시 계층, 학습/추론 방식을 바꿀 때 나머지를 얼마나 건드려야 하는지 본다.
10. 임시 해결보다 확장 가능한 해결을 우선한다.

- 단, 과설계는 피한다.

11. 경계에서는 canonical representation을 우선한다.

- 같은 의미의 데이터가 경로마다 다른 shape로 흐르지 않게 한다.
- 정규화가 필요하면 한 군데에서만 수행하고, 나머지는 canonical shape를 사용한다.

12. 호환성 계층은 명시적으로 격리한다.

- legacy format, 임시 변환, 하위 호환 로직은 핵심 경로에 섞지 않는다.
- compatibility는 별도 계층 또는 명시적 adapter로 두고, 제거 조건도 함께 남긴다.

13. policy와 mechanism을 분리한다.

- 무엇을 할지 결정하는 규칙과, 그것을 어떻게 실행하는지는 분리한다.

14. 튜닝보다 먼저 관측 가능성을 만든다.

- 파라미터를 조정하기 전에 어떤 단계에서 실패하는지 dump, metric, trace, summary로 먼저 보이게 만든다.

15. producer와 consumer를 함께 설계한다.

- 데이터를 만드는 쪽과 읽는 쪽이 서로 다른 가정을 갖지 않게 한다.

16. 공통 계층의 변경은 drift와 blast radius를 함께 본다.

- 문맥별 편향이 공통 계층으로 새어 들어가면 지역 계층 분리를 우선 검토한다.

17. 설정도 계약처럼 다룬다.

- 중요한 config는 typed structure와 명확한 source of truth를 가진다.

18. 리팩터링은 끝점까지 맞춘다.

- contract, producer, consumer, test, 문서까지 한 흐름으로 닫는다.

19. 검증 가능한 구조를 선호한다.

- 좋은 구조는 테스트, dump, compose, lint, 실행 결과로 검증 가능해야 한다.

20. 임시 단일 구현 후 재리팩토링 금지
   - “같은 family의 후속 알고리즘이 예상되면 첫 구현부터 family-level 확장면을 먼저 만든다.”

21. 외부 알고리즘 port 시 source traceability 유지
   - “논문/외부 repo 알고리즘을 가져올 때는 핵심 수식, 하이퍼파라미터 이름, 단계 구조를 원본과 최대한 1:1로 유지하고, 달라진 부분은 명시적으로 기록한다.”

22. 무의식적 근사 금지
   - “원본 알고리즘의 전제조건이 빠져 있으면 조용히 proxy 동작으로 대체하지 말고, 빠진 전제와 현재 근사
      수준을 먼저 드러낸다.”

23. family runner 우선
   - “runner/helper는 알고리즘별 일회성 파일보다 family 공통 runner + algorithm-specific adapter 구조를 우선한다. 단, 후속 알고리즘이 없을 때만 one-off를 허용한다.”
   
24. 실험 의미와 코드 완료를 분리
   - “코드가 돌아간다와 실험 비교선으로 충분히 닫혔다를 같은 완료로 취급하지 않는다. 입력 생성,
      provenance, diagnostics까지 있어야 실험 완료로 본다.”

## 패턴 선택 기준

패턴 자체를 목적처럼 쓰지 않는다. 아래 기준은 출발점일 뿐이며, 필요하면 조합해서 쓴다.

- 알고리즘만 바뀌면 `Strategy`
- 생성 조합이 바뀌면 `Factory` 계열
- 상태 단계가 중요하면 `State`
- 규칙 판단이 핵심이면 `Policy` 또는 `Specification`
- 처리 흐름이 핵심이면 `Pipeline`
- 외부 시스템 교체가 핵심이면 `Port/Adapter`
- 횡단 기능 추가면 `Decorator`
- 단순 구현체 조회면 `Registry`

## 공통 계층 vs 문맥별 계층 판단 기준

### 공통 계층에 둘 것

- 여러 문맥에서 안정적으로 유지된다
- 합쳐도 의미 왜곡이 작다
- 잘못돼도 영향 반경이 상대적으로 작다

### 문맥별 계층에 둘 것

- 사용자, 환경, 기능별로 해석 차이가 크다
- 초기 편향이 공통 계층으로 퍼지면 위험하다
- 민감한 상태나 개인화된 해석을 가진다

## 문서 규칙

- 필드 의미는 계약 파일에 직접 드러나야 한다.
- 별도 문서는 설계 이유, 배경, 예외 규칙만 둔다.
- 같은 배경 설명과 계획을 여러 긴 문서에 반복하지 않는다.

## instruction 계층화 규칙

repo instruction은 계층적으로 둔다.

1. repo-wide 공통 지침 1장
2. 필요한 경로에만 path-specific 지침
3. 개인 취향은 repo 밖 전역 레이어

## 결과물 체크리스트

- 바뀔 축이 코드에서 독립적으로 보이는가
- 공통 계층과 문맥별 계층이 섞이지 않는가
- 계약 파일만 읽어도 필드 의미가 이해되는가
- repo-wide 지침과 path-specific 지침이 충돌하지 않는가
- canonical representation이 경계에서 유지되는가
- policy와 mechanism이 분리돼 있는가
- producer와 consumer가 같은 계약을 보고 있는가
- compatibility가 핵심 경로에 새지 않는가
- 구조 변경의 검증 흔적이 남아 있는가
- 새 implementation 추가 시 기존 구현을 뜯지 않고 확장 가능한가

필요하면 [instruction_layers.md](../../.codex/skills/architecture-first-modular-engineering/references/instruction_layers.md)를 읽고,
도구별 적용 형식을 고른다.
