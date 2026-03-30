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
