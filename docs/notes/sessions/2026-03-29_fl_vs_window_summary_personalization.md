# 2026-03-29 FL vs WindowSummary Personalization 논의 요약

이 파일은 archive-only 세션 요약이다. 현재 source of truth는
`docs/project_execution_plan.md`, `docs/architecture/system-overview.md`,
`docs/fl_runtime_implementation_checklist.md`다.

## 핵심 결정

- `WindowSummary`, `NormPack`을 서버로 올리는 analytics 중심 구조는 활성 경로에서 제외했다.
- 목표는 원문 텍스트와 개인 해석 상태를 agent-local boundary에 남기고, 서버에는 공유 가능한 update/artifact만 보내는 FL 구조다.
- 초기 seed는 중앙집중형 `fixed embedding + classifier`로 만들고, 중앙 SSL은 pooled/offline control로만 해석한다.
- 논문 메인 비교는 `FL SSL under non-IID`에서 수행한다.
- 개인화는 로컬 threshold, baseline, time-series accumulator, decision policy에서 처리한다.
- prototype은 bootstrap/comparison/reference artifact로 유지하지만 메인 판정기를 prototype-only로 고정하지 않는다.

## 남긴 질문

- 어떤 adapter family를 FL runtime v1에서 열 것인가.
- query buffer raw text retention 기본값을 어떻게 둘 것인가.
- secure aggregation/DP를 어느 단계에서 도입할 것인가.

## 현재 반영 위치

- 활성 계획: `docs/project_execution_plan.md`
- 시스템 경계: `docs/architecture/system-overview.md`
- FL 구현 체크리스트: `docs/fl_runtime_implementation_checklist.md`
- contract source of truth: `shared/src/contracts/*`, `shared/src/domain/entities/*`
