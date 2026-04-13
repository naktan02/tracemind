# 2026-04-13 sandbox pytest fallback for query adaptation refactor

## 배경

- `query adaptation` 리팩터링 검증 중
  `uv run pytest ...`를 sandbox 안에서 먼저 실행했다.
- `uv` cache 경로 쓰기 제한 때문에 sandbox에서 바로 실패했다.

## 관찰한 실패

- 오류 예시:
  `failed to open file /home/jmgjmg102/.cache/uv/...: Read-only file system`

## 운영 규칙

- TraceMind repo에서 `uv run pytest` 같은 검증 명령이 sandbox 쓰기 제한으로 실패하면,
  같은 명령을 sandbox 밖 실행 승인으로 즉시 다시 시도한다.
- 특히 `uv` cache, wheel build, 임시 파일 생성 때문에 실패한 경우
  sandbox 내부 우회보다 sandbox 밖 재실행을 우선한다.

## 적용 범위

- 이번 사례의 직접 대상:
  - `scripts/experiments/lora_classifier/`
  - `agent/src/services/training/query_adaptation/`
- 같은 패턴의 `uv run pytest`, `uv run python` 검증에도 동일하게 적용한다.

## 메모

- 이 문서는 incident/archive 메모다.
- source of truth는 repo instruction과 실행 규칙 문서에 둔다.
