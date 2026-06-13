# Captured Text Repository

이 package는 `features/captured_text/` 전용 agent-local SQLite 저장소를 소유한다.

읽기 시작점:

- `repository.py`: 외부 caller가 쓰는 public repository API.
- `records.py`: 저장소 record value object와 payload 정규화.
- `schema.py`: SQLite schema와 legacy reset.
- `event_store.py`: `captured_text_events` raw event table.
- `view_job_store.py`: view generation job 상태 table.
- `generated_view_store.py`: generated weak/strong view와 training source query.
- `analysis_job_store.py`: weak text analysis job 상태 table.
- `retention.py`: retention/capacity purge와 orphan cleanup.

feature 밖 caller는 `repository.py`와 `records.py`만 직접 import하고, table별 SQL은
store module 내부에 둔다. SQLite connection과 기본 DB path mechanism은 공통
`agent/src/infrastructure/repositories/local_agent_database.py`를 사용한다.
