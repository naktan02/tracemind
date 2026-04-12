# 2026-03-31 Architecture Style Instructions Update

## User

내 스타일 알겠지? 밑의 규칙에서 추가할거 있나?

## Assistant

알고 있습니다. 네 스타일은 단순한 “모듈화 선호”가 아니라 변화 축 분리, 계약 우선, source-adjacent 문서화, canonical shape, producer/consumer 정합성, 관측 가능성 우선, 끝점까지 닫는 리팩터링 쪽이다.  
그래서 `canonical representation`, `compatibility 격리`, `policy와 mechanism 분리`, `관측 가능성 먼저`, `producer/consumer 함께 설계`, `공통 계층 drift와 blast radius`, `config도 계약처럼`, `리팩터링은 끝점까지`, `검증 가능한 구조` 원칙을 추가 제안했다.

## User

최종판 만드는데 이 프로젝트에 국한되게 만든거 아니지?  
그리고 compatibility는 별도 계층 또는 명시적 adapter로 두고, 제거 조건도 남긴다. 이건 무슨 말이야?

## Assistant

최종판은 특정 프로젝트 용어 없이 일반화해서 다시 정리했다.  
`compatibility를 별도 계층으로 둔다`는 건 legacy format, 임시 변환, 하위 호환 로직을 핵심 비즈니스 경로에 흩뿌리지 말고, 별도 serializer/adapter/compat layer에 모아두라는 뜻이다.  
`제거 조건도 남긴다`는 건 이 호환 코드가 언제 필요 없어지는지, 예를 들어 “모든 producer가 새 포맷으로 전환되면 제거” 같은 종료 조건을 남기라는 뜻이다.

## User

이제 이거대로 skills나 에이전트에 적어두었던거 이전 버전 남아있으면 이걸로 최신화진행해

## Assistant

저장소 안의 공통 지침 파일들에서 이전 버전 문구를 찾아 새 최종 원칙으로 맞추기로 했다.

## Changes

- `.codex/skills/architecture-first-modular-engineering/SKILL.md`
  - 일반화된 최종 원칙으로 확장
  - `canonical representation`
  - `compatibility 격리`
  - `policy/mechanism 분리`
  - `관측 가능성`
  - `producer/consumer 정합성`
  - `drift/blast radius`
  - `config as contract`
  - `end-to-end refactoring`
  - `검증 가능한 구조`
  반영
- `.codex/skills/architecture-first-modular-engineering/agents/openai.yaml`
  - short description과 default prompt를 최신 원칙에 맞게 수정
- `.github/copilot-instructions.md`
  - repo-wide 지침에 canonical representation, compatibility layer, policy/mechanism 분리, observability, producer/consumer alignment 추가
- `.github/instructions/architecture.instructions.md`
  - path-specific 지침에 위 원칙들을 요약 반영
- `AGENTS.md`
  - `Architecture Direction` 문단을 최신 철학에 맞게 갱신

## Notes

- 이번 변경은 특정 프로젝트 전용 도메인 용어를 넣지 않고, 어떤 프로젝트에서도 재사용 가능한 일반 원칙으로 유지했다.
- 저장소 특화 규칙은 `AGENTS.md`의 기존 repo 운영 규칙으로 남기고, 아키텍처 철학 부분만 최신화했다.

## User

tracemind_project skills 뭐라고 적혀 있는지 알려줘

## Assistant

`tracemind-project` 스킬에는 대체로 아래 규칙이 적혀 있다고 설명했다.

- 항상 한국어로 답변
- 중요한 작업 전에 계획 문서 확인
- `docs/notes/`는 기본적으로 읽지 않음
- 임시방편보다 미래지향적 구조 우선
- 도메인과 계약 우선 설계
- 현재 세션 대화 전문은 `docs/notes/` 아래에 남기는 것이 기본 규칙
- `plan.md`는 비전/연구 목적
- `docs/project_execution_plan.md`는 구현 순서와 활성 아키텍처

## User

지금 밑의 글에다가 docs/notes/는 기본적으로 읽지 않음, 항상 한국어로 답변, 중요한 작업 전에 계획 문서부터 확인,현재 세션 대화 전문은 docs/notes/ 아래에 남기는 것이 기본 규칙, plan.md는 비전/연구 목적, docs/project_execution_plan.md이라는거 덧붙여서 수정진행해

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

필요하면 [instruction_layers.md](../../../.codex/skills/architecture-first-modular-engineering/references/instruction_layers.md)를 읽고,
도구별 적용 형식을 고른다.

## Assistant

일반화된 `Architecture-First Modular Engineering` 스킬에 TraceMind 저장소 전용 운영 규칙을 추가하는 방향으로 수정하기로 했다.

## Changes

- `.codex/skills/architecture-first-modular-engineering/SKILL.md`
  - `TraceMind 저장소에서 추가로 지킬 것` 섹션 추가
  - 한국어 응답 규칙 반영
  - `plan.md`, `docs/project_execution_plan.md` 우선 확인 규칙 반영
  - `plan.md`와 `docs/project_execution_plan.md`의 역할 구분 반영
  - `docs/notes/` 기본 비열람과 예외 조건 반영
  - 현재 세션 대화 전문 기록 규칙 반영
- `.codex/skills/architecture-first-modular-engineering/agents/openai.yaml`
  - short description과 default prompt를 TraceMind 운영 규칙까지 포함하도록 갱신

## User

지금 수정한 내용이 TraceMind Project TraceMind 프로젝트 전용 작업 흐름과 규칙 안내 에 들어간 내용 맞지?

## Assistant

아니고, 방금 수정한 내용은 `TraceMind Project` 스킬이 아니라 `Architecture-First Modular Engineering` 스킬에 들어갔다고 설명했다.

## User

TraceMind Project TraceMind 프로젝트 전용 작업 흐름과 규칙 안내를 수정하라는 말이었어

## Assistant

의도를 잘못 해석한 것을 인정하고, 실제 대상인 `tracemind-project` 스킬을 수정하고 잘못 넣었던 TraceMind 전용 문구는 `architecture-first-modular-engineering` 스킬에서 제거하기로 했다.
