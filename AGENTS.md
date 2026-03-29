# Repository Guidelines

## Project Structure & Module Organization
Keep runtime code inside `src/`, grouped by responsibility: `src/api` for FastAPI routers, `src/core` for shared domain models, and `src/services` for background or external integrations. Unit and integration tests mirror the package layout inside `tests/`. Reusable task runners or data seeding scripts live in `scripts/`, while deployment manifests (Dockerfiles, Terraform, GitHub Actions) belong in `infra/`.
실행용 스크립트 설정은 `scripts/conf/`의 Hydra config group를 source of truth로 유지한다. `dataset`, `embedding`, `runtime` group를 기본 축으로 사용하고, 스크립트의 기본 runtime은 `gpu_online`으로 본다. 스크립트용 preset을 Python helper 파일에 다시 복제하지 않는다.

## Build, Test, and Development Commands
Create a virtual environment (`python -m venv .venv && source .venv/bin/activate`) and install dependencies with `pip install -r requirements.txt`. `uvicorn src.api.main:app --reload` starts the local API server with auto-reload. Run `pytest` for the default test suite. Use `ruff check src tests` before pushing to catch lint violations, and `black src tests` if formatting drifts. Container workflows should rely on `docker compose up api` to ensure parity with production services.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and Black-compatible line wrapping (88 chars). Prefer dataclasses or Pydantic models for request/response bodies. Use descriptive snake_case for modules, functions, and variables, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants. Annotate public functions with explicit type hints.
Write comments and explanatory prose in Korean unless a path, code identifier, third-party model name, or protocol term needs to stay in its original language.

## Testing Guidelines
Pytest is the authoritative framework. Name files `test_<module>.py` and functions `test_<behavior>`. Keep fast, deterministic cases in `tests/unit`, while contract and integration checks belong in `tests/integration` and may use the `slow` marker. Aim for ≥90% statement coverage on core services; fail the pipeline if coverage drops below the threshold by using `pytest --cov=src --cov-fail-under=90`. Use factories/fixtures to build domain objects instead of manual dictionaries to reduce brittleness.

## Commit & Pull Request Guidelines
Write commits in the imperative mood (`feat: add session cache`) and keep them scoped to a single concern. Reference an issue ID in the body when applicable, and include test evidence (`pytest -m 'not slow'`) if behavior changes. Pull requests should describe the motivation, summarize architectural decisions, and attach screenshots or API samples when the change is user-facing. Ensure CI is green, link design docs when touching shared interfaces, and request reviews from both backend and infra maintainers if changes span modules.

## Security & Configuration Tips
Never commit `.env` files; instead, maintain `.env.example` with placeholder values. Rotate API keys quarterly and prefer environment variables over hard-coded secrets. If a migration touches personally identifiable data, document the retention policy in the PR and obtain approval before deploying.
