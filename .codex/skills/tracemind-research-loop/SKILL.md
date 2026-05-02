---
name: tracemind-research-loop
description: Thin TraceMind router for experiment intent alignment, method comparison, bug fixes, feature work, refactors, E2E tests, or harness maintenance.
---

# TraceMind Research Loop

이 skill은 TraceMind에서 어떤 project skill을 언제 쓸지 고르는 얇은 router다.
원본 skill 절차는 재정의하지 않는다.

## First Read

필요한 만큼만 읽는다.

1. `docs/ai_context_manifest.yaml`
2. `docs/execution_index.md`
3. `AGENTS.md`
4. 관련 path-specific `AGENTS.md`
5. 관련 contract, Hydra config, code, test

## GRILL Rule

알고리즘, 방법론, baseline, ablation, paper-track 비교는 구현이나 실행 전에
`grill-with-docs` 방식으로 정렬한다.

- 계획의 모든 측면을 집요하게 질문해 shared understanding에 도달한다.
- design tree의 각 branch를 따라가며 결정 사이의 의존성을 하나씩 해소한다.
- 질문은 한 번에 하나씩 한다.
- 코드와 문서로 답할 수 있으면 먼저 확인한다.
- 권장 답을 함께 제시한다.
- 최소한 목표, 고정 변수, 변경 변수, dataset/split/seed, metric, output metadata,
  승격 기준을 명시한다.
- 확정된 의미는 `plan.md`, `docs/contracts/*`, code-adjacent README 같은 기존
  TraceMind source of truth에 반영한다. 새 `CONTEXT.md`나 ADR은 기본값이 아니다.

## Routing

- 실험 의도, 용어, 비교 기준 stress test: `grill-with-docs`
- 버그, 실패, 회귀, 성능 문제: `diagnose`
- test-first 기능/수정: `tdd`
- 낯선 코드 큰 그림: `zoom-out`
- 구조 개선/분리/삭제 후보: `improve-codebase-architecture`
- TraceMind contract 변경: `tracemind-contract-sync`
- 새 strategy/adapter/backend: `tracemind-strategy-addition`
- FL round/artifact drift/revision mismatch: `tracemind-federation-debug`
- active docs drift 정리: `tracemind-doc-sync`

## TraceMind Constraints

- 한국어 응답, 주석, 설명 문서를 기본으로 한다.
- `shared/src/contracts/`와 `shared/src/domain/entities/`가 계약 source of truth다.
- 실험 설정 source of truth는 `scripts/conf/` Hydra config group이다.
- seed baseline, central SSL control, FL SSL non-IID main comparison, runtime translation을 섞지 않는다.
- Web/app은 contract/API consumer이며 도메인 의미를 재정의하지 않는다.
- Codex subagent는 사용자가 명시적으로 요청한 경우에만 쓴다.
- 변경은 테스트, lint, 실행 결과, docs 동기화 중 적절한 검증으로 닫는다.

## Upstream

`diagnose`, `tdd`, `grill-with-docs`, `zoom-out`,
`improve-codebase-architecture`는 Matt Pocock의 `skills` repository에서 가져온
workflow를 가능한 원형으로 보존한다. 출처와 license는
`.codex/skills/THIRD_PARTY_NOTICES.md`를 본다.
