# TraceMind Document Governance

이 문서는 TraceMind 문서 운영 규칙의 canonical source of truth다.

목표는 live 구현, 활성 연구 계획, historical notes를 분리해서 AI와 사람이 같은 source-of-truth 순서를 따르게 하는 것이다.

## 1. 문서 클래스

| 분류 | 의미 | 예시 |
|---|---|---|
| Code-adjacent Truth | payload 필드, domain entity, canonical helper의 최종 정본 | `shared/src/contracts/*.py`, `shared/src/domain/entities/*`, `shared/src/contracts/README.md` |
| Canonical Active Docs | 현재 구현, 현재 계획, 운영 경계를 설명하는 공식 문서 | `README.md`, `docs/execution_index.md`, `docs/architecture/system-overview.md` |
| Contract Design Docs | contract 설계 배경과 확장 절차 | `docs/contracts/*` |
| Operational Docs | 로컬 실행, smoke, test, runbook | `docs/operations/*`, `docs/quality/*` |
| Harness Docs | AI/Codex 작업 라우팅과 eval case | `docs/ai_context_manifest.yaml`, `docs/ai_harness_*` |
| Notes | 세션, 사건, 결정 아카이브 | `docs/notes/**` |
| Mockups | UI/제품 실험용 정적 산출물 | `docs/mockups/**` |

## 2. Source of Truth 표

| 주제 | 정본 |
|---|---|
| 저장소 진입과 빠른 요약 | `README.md` |
| AI 작업 라우팅 | `docs/ai_context_manifest.yaml` |
| 읽기 순서와 문서 지도 | `docs/execution_index.md` |
| 시스템 구조와 코드 경계 | `docs/architecture/system-overview.md` |
| 활성 연구/시스템 계획 | `docs/project_execution_plan.md` |
| staged phase map | `docs/staged_execution_roadmap.md` |
| API endpoint 표면 | `docs/api/api-surface.md`, `agent/src/api/*`, `main_server/src/api/*` |
| payload 필드와 의미 | `shared/src/contracts/*.py`, `shared/src/contracts/README.md` |
| local run/smoke 절차 | `docs/operations/local-runbook.md` |
| 테스트 전략 | `docs/quality/test-strategy.md` |
| 알고리즘/전략 추가 절차 | `docs/strategy_surface_map.md`, `docs/contracts/strategy_addition_playbook.md`, 필요 시 `docs/contracts/algorithm_extension_guide.md` |
| query buffer local boundary | `docs/contracts/query_buffer_v1.md`, `agent/src/infrastructure/repositories/query_buffer_repository.py` |
| child-support local boundary | `shared/src/contracts/child_support_contracts.py`, `agent/src/services/wellbeing/child_support_service.py` |
| family extension planning | `docs/family_extension_wellbeing_signal_mvp_plan.md`, `shared/src/contracts/wellbeing_signal_contracts.py` |

## 3. 작성 원칙

1. 필드 의미와 payload shape는 가능하면 contract 코드 가까이에 둔다.
2. active docs는 현재 구현과 현재 우선순위만 설명한다.
3. planned 항목은 planned/future라고 명시하거나 roadmap/plan 문서에 둔다.
4. notes는 historical context이며 active guidance로 취급하지 않는다.
5. 문서 하나는 목적 하나를 가진다.
6. 같은 배경 설명을 여러 문서에 길게 복제하지 않는다.
7. API 문서는 field 전체 복제보다 route, owner, contract source를 명시한다.
8. 운영 절차는 실제 존재하는 command, config, 파일 기준으로 쓴다.
9. 읽기 안내 문서는 전체 읽기 순서가 아니라 task route와 필요한 문서 선택 기준을 제공한다.
10. 세션 기록은 기본적으로 300-500 words 요약만 남기고, 대화 전문 transcript는 repo 안 active/archive 문서로 추가하지 않는다.

## 4. 언제 어떤 문서를 갱신하는가

| 변화 | 갱신 문서 |
|---|---|
| top-level ownership, runtime rail 변경 | `README.md`, `docs/architecture/system-overview.md`, `docs/execution_index.md` |
| task route, source-of-truth 우선순위 변경 | `docs/ai_context_manifest.yaml`, `docs/execution_index.md` |
| API route 추가/삭제/의미 변경 | `docs/api/api-surface.md`, 관련 package tests |
| shared payload 변경 | contract file, `shared/src/contracts/README.md`, 관련 `docs/contracts/*`, tests |
| local run command, dependency, GPU/runtime profile 변경 | `docs/operations/local-runbook.md`, `README.md`, 필요 시 `AGENTS.md` |
| 테스트 구조나 quality gate 변경 | `docs/quality/test-strategy.md`, `tests/AGENTS.md` |
| script/Hydra config source 변경 | `scripts/README.md`, `docs/execution_index.md`, 관련 tests |
| UI consumer contract 변경 | app code, generated type tests, `docs/api/api-surface.md` |
| 의미 있는 설계 결정 | `docs/notes/decisions/YYYY-MM-DD-*.md`, 필요 시 active docs 요약 |
| incident 또는 flaky/debug 절차 | `docs/notes/incidents/YYYY-MM-DD-*.md`, 필요 시 runbook 요약 |

## 5. Live / Planned / Historical 구분

### Live

- 코드, API, script, test, app 화면이 현재 존재한다.
- 실행 절차나 owner를 설명할 수 있다.
- canonical active docs에 포함 가능하다.

### Planned

- 설계 방향, future option, placeholder다.
- active docs에 둘 경우 planned/future임을 명시한다.
- 구현된 것처럼 API/runbook에 넣지 않는다.

### Historical

- 현재 정본은 아니지만 과거 판단을 이해하는 데 가치가 있다.
- `docs/notes/**` 또는 별도 archive 성격 문서에 둔다.
- active guide의 근거로 직접 사용하지 않는다.

## 6. 품질 게이트

문서 변경 전후 아래를 확인한다.

1. 실제 파일이나 endpoint가 존재하는가.
2. command가 현재 dependency source와 일치하는가.
3. live와 planned 설명이 섞이지 않았는가.
4. 같은 source-of-truth가 두 문서에서 서로 다르게 설명되지 않는가.
5. contract 의미를 docs가 코드 대신 소유하지 않는가.
6. `docs/execution_index.md`와 `docs/ai_context_manifest.yaml`이 새 문서를 가리키는가.

## 7. 파일명 규칙

- 소문자 kebab-case를 기본으로 쓴다.
- 날짜가 필요한 notes는 `YYYY-MM-DD-topic.md`를 쓴다.
- `final`, `misc`, `temp`, `new`, `copy` 같은 이름은 피한다.

## 8. Notes 사용 규칙

`docs/notes/**`는 active source of truth가 아니다.

| 하위 경로 | 용도 |
|---|---|
| `docs/notes/decisions/` | 대안 비교와 선택 이유 |
| `docs/notes/incidents/` | 장애, sandbox 이슈, flaky/debug 기록 |
| `docs/notes/sessions/` | 짧은 세션 요약과 작업 맥락 |

notes에 남긴 내용이 현재 규칙이 되면 active docs나 contract 코드 가까이 요약해서 승격한다.
기존에 남아 있는 긴 session transcript는 historical archive로만 취급하고 통째로 읽지 않는다.
