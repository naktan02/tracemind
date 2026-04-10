# Tests AGENTS

## 역할

repo 루트 `tests/`는 cross-boundary integration, e2e, architecture 검증을
소유한다.

## 먼저 읽을 것

1. 루트 `AGENTS.md`
2. 관련 소유 경계의 `AGENTS.md`
3. 작업과 직접 맞닿은 기존 테스트 파일

## 변경 규칙

- 패키지 내부 단위 테스트는 각 패키지 아래 테스트 디렉터리에 둔다.
- root `tests/`에는 경계를 넘는 시나리오만 둔다.
- contract 변경은 serialization/compatibility 검증과 함께 닫는다.
- architecture 규칙 변경은 `tests/architecture/` 또는 동등한 구조 검증으로
  드러나야 한다.
- flaky한 외부 의존 대신 결정적인 fixture와 sample payload를 우선한다.
