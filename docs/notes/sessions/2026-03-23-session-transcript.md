# 2026-03-23 monorepo 구조 논의 요약

이 파일은 archive-only 세션 요약이다. 초기 대화 전문은 현재 구조와 맞지 않는
오래된 경로를 포함하므로 active rule로 사용하지 않는다.

## 핵심 결정

- 중앙 서버와 로컬 에이전트는 당장 별도 repo로 나누지 않고 monorepo로 유지한다.
- 물리 repo 분리보다 shared contract와 경계 안정화를 우선한다.
- 배포 주기, 팀 경계, CI/release 요구가 실제로 분리될 때 repo split을 다시 판단한다.
- 현재 구조는 `shared`, `methods`, `agent`, `main_server`, `scripts`, `apps`, `tests`, `docs`로 역할을 나눈다.

## 현재 기준

- 공통 계약과 domain entity는 `shared/`.
- 교체 가능한 알고리즘/method 계산 core는 `methods/`.
- 로컬 inference/training과 private state는 `agent/`.
- round lifecycle, aggregation, publication은 `main_server/`.
- Hydra 실험 entrypoint와 report는 `scripts/`.
- UI shell과 API consumer는 `apps/`.

## 현재 반영 위치

- 최상위 규칙: `AGENTS.md`
- 실행 계획: `docs/project_execution_plan.md`
- 시스템 개요: `docs/architecture/system-overview.md`
- 문서 거버넌스: `docs/AGENTS.md`, `docs/governance/document-governance.md`
