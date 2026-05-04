# 2026-04-01 output storage layout와 runs 구조 요약

이 파일은 archive-only 세션 요약이다. 현재 산출물 위치 기준은
`docs/architecture/system-overview.md`, `docs/operations/local-runbook.md`,
`scripts/README.md`를 따른다.

## 핵심 결정

- `data/`는 재사용 가능한 입력과 가공 산출물을 둔다.
- `runs/`는 실험 1회 실행별 결과, report, diagnostics, visualization, log를 둔다.
- `agent/state/`는 agent runtime이 실제로 읽고 쓰는 local/private state를 둔다.
- `main_server/state/`는 server runtime state, publication artifact, manifest를 둔다.
- `evaluations/`처럼 의미가 섞이는 top-level 폴더는 새로 만들지 않는다.
- 루트에 `main.log`, ad-hoc simulation output이 쌓이지 않게 entrypoint는 `runs/<job>/<run_id>/`를 사용한다.

## 현재 구조 기준

```text
data/
  raw/
  processed/
runs/
  <job>/<run_id>/
agent/state/
main_server/state/
hf_cache/
```

## 남은 주의점

- 재사용 artifact와 실행 결과를 섞지 않는다.
- runtime state를 논문 report archive처럼 쓰지 않는다.
- 실행별 날짜/시간은 `run_id`와 report metadata에 남긴다.
