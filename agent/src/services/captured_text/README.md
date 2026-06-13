# Captured Text Services

`captured_text/`는 agent-local raw text ingest부터 generated view, debug analysis job,
training source projection까지의 lifecycle을 소유한다.

읽기 시작점:

- `ingest.py`: `CapturedTextEventPayload`를 raw event store에 저장한다.
- `lifecycle.py`: retention과 max-record purge 정책을 실행한다.
- `view_generation/`: raw event를 weak/strong generated view로 materialize한다.
- `debug_jobs.py`: view generation과 weak text analysis를 한 번의 debug run으로 묶는다.
- `training_source/`: ready generated view를 Query SSL unlabeled row로 정규화한다.

raw text는 repository 경계에 남기고, training backend에는 generated weak/strong view를
정규화한 row만 넘긴다.
