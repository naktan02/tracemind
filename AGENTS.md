# Repository Guidelines

## AI Harness Entry Points

이 저장소는 Codex CLI와 VS Code Codex extension이 같은 하네스를 공유하도록
구성한다. 작업 시작 기본 순서는 아래와 같다.

1. `docs/ai_context_manifest.yaml`
   - task별 읽기 순서와 source-of-truth 우선순위
2. `docs/execution_index.md`
   - 짧은 문서 지도
3. 현재 작업 경로의 path-specific `AGENTS.md`
   - `shared/`, `agent/`, `main_server/`, `scripts/`, `tests/`, `docs/`
4. 관련 code contract와 active docs

## Project Structure & Ownership

이 저장소는 역할별 모듈이 나뉜 monorepo 구조다.

- `shared/`
  - 공통 contract, domain entity, canonical 계산 규칙의 source of truth
- `agent/`
  - agent-owned 로컬 추론/로컬 학습 runtime과 API
- `main_server/`
  - server-owned round lifecycle, aggregation, publication orchestration과 API
- `scripts/`
  - 코어를 조합하는 실험층
  - thin wrapper, sweep, report, visualization, exploratory-only logic만 둔다
- `tests/`
  - cross-boundary integration/e2e 및 architecture 검증
- `infra/`
  - 배포/운영 manifest

중요:

- 운영 후보 알고리즘 구현은 `scripts`에 먼저 만들고 나중에 복사하지 않는다.
- 공용 계산 규칙은 `shared`, agent-owned 로컬 실행은 `agent`,
  server-owned orchestration은 `main_server`에 둔다.
- 테스트 위치도 책임 경계를 따른다. 패키지별 테스트는 각 패키지 아래에 두고,
  경계를 넘는 검증만 repo 루트 `tests/`에 둔다.
- 실행용 스크립트 설정은 `scripts/conf/`의 Hydra config group를 source of
  truth로 유지한다. `dataset`, `embedding`, `runtime` group를 기본 축으로
  사용하고, 스크립트의 기본 runtime은 `gpu_online`으로 본다.

## Documentation Priority

문서 우선순위는 다음과 같이 본다.

1. `shared/src/contracts/*.py`, `shared/src/domain/entities/*`
2. `shared/src/contracts/README.md`
3. `docs/contracts/*`, `docs/*`
4. `docs/notes/*`

기본 진입점은 아래 순서를 따른다.

1. `docs/ai_context_manifest.yaml`
2. `docs/execution_index.md`
3. `plan.md`
4. `docs/project_execution_plan.md`
5. `docs/contracts/central_lora_classifier_trainer_contract.md`
   - 논문 트랙의 canonical LoRA scaffold와 산출물 경계
6. `shared/src/contracts/README.md`
7. `docs/fl_runtime_implementation_checklist.md`
8. `docs/contracts/algorithm_extension_guide.md`

## Architecture Direction

구조 변경은 contract-first와 change-axis separation을 우선한다.

- 서로 다른 이유로 바뀌는 책임을 한 클래스나 한 payload에 섞지 않는다.
- `shared/src/contracts/`와 `shared/src/domain/entities/`를 source of truth로
  본다.
- 중요한 필드 의미는 코드 가까이에 둔다.
- 경계에서는 canonical representation을 우선한다.
- legacy format이나 임시 변환은 compatibility 계층으로 명시적으로 격리한다.
- 공통 계층과 문맥별 계층을 분리한다.
- policy와 mechanism을 분리한다.
- 튜닝 전에 dump, trace, summary 같은 관측 가능성을 먼저 만든다.
- raw registry는 얇은 wiring 용도로만 쓰고 핵심 도메인 추상화로 남용하지
  않는다.

## Build, Test, and Development Commands

- 환경 준비: `python -m venv .venv && source .venv/bin/activate`
- 의존성 설치: `pip install -r requirements.txt`
- API 서버: `uvicorn main_server.src.api.main:app --reload`
- 기본 테스트: `pytest`
- lint: `ruff check main_server/src agent/src shared/src tests`
- format: `black main_server/src agent/src shared/src tests`
- 컨테이너 parity: `docker compose up api`

테스트나 디버그 실행이 중단됐거나 터미널 상태가 이상하면 먼저 `ps aux`를
실행해 남아 있는 `pytest`, `uv run python`, 기타 stale 프로세스를 확인한다.

## Coding Style & Naming Conventions

- Python 3.11 기준
- four-space indentation
- Black-compatible line wrapping
- public 함수는 명시적 type hint
- 모듈/함수/변수는 snake_case
- 클래스는 PascalCase
- 상수는 SCREAMING_SNAKE_CASE
- request/response body는 dataclass 또는 Pydantic 우선
- 주석과 설명 문서는 기본적으로 한국어로 쓴다

## Testing Guidelines

- Pytest가 authoritative framework다.
- 파일명은 `test_<module>.py`, 함수명은 `test_<behavior>`를 쓴다.
- 빠른 deterministic 검증은 `tests/unit`
- contract/integration 검증은 `tests/integration`
- core service는 statement coverage 90% 이상을 목표로 본다.
- 수동 dict보다 fixture/factory를 우선한다.

## Commit & Pull Request Guidelines

- commit message는 imperative mood로 쓴다. 예: `feat: add session cache`
- 한 commit은 한 concern에 집중한다.
- behavior 변경 시 테스트 근거를 남긴다.
- shared interface를 건드리면 관련 설계 문서를 함께 연결한다.

## Security & Configuration Tips

- `.env`는 커밋하지 않는다. 대신 `.env.example`을 유지한다.
- 비밀값은 하드코딩보다 environment variable을 우선한다.
- PII가 관련된 migration은 retention policy를 문서화하고 승인 후 배포한다.
