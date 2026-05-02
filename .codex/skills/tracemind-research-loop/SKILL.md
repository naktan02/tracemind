---
name: tracemind-research-loop
description: Thin TraceMind router for aligning research intent and choosing project skills. Use when starting TraceMind planning, paper-track experiments, algorithm or method comparison, ambiguous experiment language, bug fixes, feature work, refactors, E2E tests, or harness maintenance.
---

# TraceMind Research Loop

이 skill은 원본 workflow를 재정의하지 않는다. TraceMind에서 어떤 skill을 언제
쓸지 연결하는 얇은 overlay다. 세부 절차는 각 skill의 `SKILL.md`를 그대로
따른다.

## First Read

필요한 만큼만 읽는다.

1. `docs/ai_context_manifest.yaml`
2. `docs/execution_index.md`
3. `AGENTS.md`
4. 관련 path-specific `AGENTS.md`
5. 관련 contract, Hydra config, code, test

## Default GRILL Rule

알고리즘, 방법론, baseline, ablation, paper-track 비교처럼 사용자의 의도와
agent의 해석이 갈릴 수 있는 작업은 구현이나 실행 전에 `grill-with-docs` 방식으로
정렬한다.

- 계획의 모든 측면을 집요하게 질문해 shared understanding에 도달한다.
- design tree의 각 branch를 따라가며 결정 사이의 의존성을 하나씩 해소한다.
- 질문은 한 번에 하나씩 한다.
- 코드와 문서로 답할 수 있는 질문은 먼저 확인한다.
- 각 질문에는 권장 답을 함께 제시한다.
- 최소한 비교 목표, 고정 변수, 변경 변수, dataset/split/seed, metric, 출력
  metadata, 승격 기준을 명시한다.
- 확정된 의미는 기존 source of truth에 맞춰 반영한다. `CONTEXT.md`나
  `docs/adr/`를 새로 만들기보다 `plan.md`, `docs/contracts/*`,
  `docs/notes/decisions/*`, 코드 가까운 README를 우선한다.

## Skill Routing

- 실험 의도, 용어, 비교 기준, 계획 stress test: `grill-with-docs`
- 버그, 실패, 회귀, 성능 문제: `diagnose`
- 새 기능, API/contract behavior, test-first 구현: `tdd`
- 낯선 코드 영역의 큰 그림 파악: `zoom-out`
- 구조 개선, deep module, deletion test, test surface 점검:
  `improve-codebase-architecture`
- TraceMind contract 변경: `tracemind-contract-sync`
- 새 전략/adapter/backend 추가: `tracemind-strategy-addition`
- FL round, artifact drift, revision mismatch 디버그: `tracemind-federation-debug`
- active docs drift 정리: `tracemind-doc-sync`

## TraceMind Constraints

원본 skill 절차보다 repo/system 지침이 우선한다.

- 사용자 응답, 주석, 설명 문서는 기본적으로 한국어로 쓴다.
- `shared/src/contracts/`와 `shared/src/domain/entities/`가 계약 source of
  truth다.
- 실험 설정의 source of truth는 `scripts/conf/` Hydra config group이다.
- seed baseline, query-domain adaptation, 시스템 FL translation을 섞지 않는다.
- baseline 비교에서는 backbone, tokenizer, LoRA spec, split, seed 같은 고정
  변수를 먼저 잠근다.
- Web/app은 contract/API consumer이며 도메인 의미를 재정의하지 않는다.
- Codex subagent는 사용자가 명시적으로 요청한 경우에만 쓴다.
- 변경은 관련 테스트, lint, 실행 결과, docs 동기화 중 적절한 검증으로 닫는다.

## Upstream

`diagnose`, `tdd`, `grill-with-docs`, `zoom-out`,
`improve-codebase-architecture`는 Matt Pocock의 `skills` repository에서 가져온
workflow를 가능한 원형으로 보존한다. 출처와 license는
`.codex/skills/THIRD_PARTY_NOTICES.md`를 본다.
