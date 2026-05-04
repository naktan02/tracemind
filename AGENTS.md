# Repository Guidelines

## Start

1. `docs/ai_context_manifest.yaml`에서 task route를 고른다.
2. `docs/execution_index.md`에서 필요한 문서만 고른다.
3. 작업 경로의 `AGENTS.md`를 읽는다.
4. 관련 contract, Hydra config, code, test를 확인한다.

Path-specific instructions: `shared/`, `agent/`, `main_server/`, `scripts/`,
`tests/`, `docs/`, `apps/`, `apps/experiment_web/`, `apps/family_extension/`.

## Working Rules

- 모호한 요구는 가정, 선택지, 성공 기준을 먼저 드러낸다.
- 알고리즘, baseline, ablation, 논문 비교는 구현 전 목표, 고정/변경 변수,
  dataset/split/seed, metric, output metadata를 맞춘다.
- 요청과 직접 연결된 파일만 고치고 주변 리팩터링은 별도 필요로 언급한다.
- 단일 사용처용 추상화, 설정 축, compatibility layer를 미리 만들지 않는다.
- 변경은 테스트, lint, 실행 결과, 문서 동기화 중 적절한 검증으로 닫는다.
- 주석과 설명 문서는 기본적으로 한국어로 쓴다.

## Ownership

- `shared/`: 공통 contract, domain entity, canonical payload 해석 규칙.
- `methods/`: 교체 가능한 알고리즘/method 계산 core.
- `agent/`: 로컬 inference/training, private/local state, FL participant.
- `main_server/`: round lifecycle, aggregation, publication, server API.
- `scripts/`: Hydra 실험 entrypoint, sweep, report, visualization thin wrapper.
- `apps/`: UI shell/API consumer. 계약 의미나 기본값을 소유하지 않는다.
- `tests/`: cross-boundary integration/e2e와 architecture guard.
- `docs/`: 설명 계층. source of truth를 대체하지 않는다.

운영 후보 알고리즘은 `scripts`에 먼저 만들고 나중에 복사하지 않는다.
재사용 알고리즘 core는 처음부터 `methods`에 둔다. 공통 contract/domain은
`shared`, runtime adapter는 `agent`/`main_server`, 실험 실행은 `scripts`에 둔다.

## Source Of Truth

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `plan.md`, `docs/contracts/*`, active `docs/*.md`

`docs/notes/**`는 archive-only다. 현재 규칙은 active docs나 code-adjacent 문서로
요약 승격한 뒤 사용한다.

## Architecture Bias

- contract-first와 change-axis separation을 우선한다.
- 필드 의미는 코드 가까이에 두고, 경계에서는 canonical representation을 쓴다.
- policy/mechanism, runtime/aggregation, privacy/training을 분리한다.
- 튜닝 전에 dump, trace, summary 같은 관측 가능성을 만든다.
- 비교 대상이 2개 이상이면 같은 경계에 다음 알고리즘을 얹을 수 있게 한다.

## Active Research

```text
central fixed embedding + classifier seed
-> central SSL pooled/offline control
-> FL SSL non-IID main comparison
-> FL/runtime translation
```

- canonical seed artifact: `clf_2026_04_11_143138`
- 중앙 SSL은 pooled/offline control이며 최종 논문 메인 랭킹이 아니다.
- `FedMatch`, `FedLGMatch`, `(FL)^2`는 non-IID client split에서 메인 비교한다.
- 원문 텍스트와 개인 해석 상태는 agent-local boundary에 남긴다.

## Commands

- install: `uv sync --extra dev --extra experiments`
- main server: `uv run uvicorn main_server.src.api.main:app --reload --port 8000`
- agent: `uv run uvicorn agent.src.api.main:app --reload --port 8001`
- tests: `uv run pytest`
- lint: `uv run ruff check main_server/src agent/src shared/src scripts tests`
- format: `uv run ruff format --check main_server/src agent/src shared/src scripts tests`

실행이 꼬이면 먼저 `ps aux`로 stale process를 확인한다. GPU 실행 전에는 실제
실행 환경에서 `nvidia-smi`와 `torch.cuda.is_available()`를 확인한다.

## Permission Boundaries

검증용 로컬 명령은 실행하되, repo 밖 대량 수정/삭제, destructive git, commit/push,
remote 변경, secret/system config 변경, 비용 API/대용량 다운로드, 무관한 프로세스
종료는 사용자 요청 또는 별도 확인 뒤 진행한다.

## Code Style

- Python 3.11, type hint, Black-compatible wrapping.
- snake_case 함수/변수, PascalCase 클래스, SCREAMING_SNAKE_CASE 상수.
- request/response body는 dataclass 또는 Pydantic 우선.
- 내부 import는 direct-file import가 기본이다. package-level export는 stable public
  API일 때만 열고, 내부 `__init__.py`는 기본적으로 marker/docstring only로 둔다.
- Pytest가 authoritative framework다.
- 빠른 deterministic 검증은 `tests/unit`, 경계 검증은 `tests/integration`에 둔다.

## Commit And Security

- commit format: `type: subject`
- `type`: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- `subject`는 한국어 가능. 예: `feat: 아이용 로컬 LLM 대화 경계 추가`
- 한 commit은 한 concern에 집중한다.
- `.env`는 커밋하지 않고 비밀값은 environment variable을 우선한다.
- PII migration은 retention policy를 문서화하고 승인 후 배포한다.
