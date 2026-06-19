# Agent

TraceMind의 로컬 runtime이다. captured text ingest, local inference, wellbeing output,
child-support conversation, local training, FL server 참여를 agent process 안에서
조립한다.

원문 텍스트, 개인 해석 상태, 개인 threshold, local training source 같은 민감한 상태는
agent-local boundary에 남긴다. `main_server`는 FL round와 aggregation을 소유하고,
`methods/`는 알고리즘 core를 소유한다.

## What It Does

- browser/extension이 보낸 captured text와 typing segment를 받는다.
- 전처리, 번역, embedding, scorer backend, local interpretation을 조합한다.
- wellbeing summary, timeseries, space-web, child-support payload를 만든다.
- family extension의 setup/unlock/session API를 제공한다.
- local training task를 실행하고 training source usage를 기록한다.
- FL server에서 받은 round task를 local update로 실행한 뒤 update payload를 업로드한다.

## Run The API

repo root에서 실행한다.

```bash
uv run uvicorn agent.src.api.main:app --reload --host 127.0.0.1 --port 8001
```

local setting이 필요하면 `agent/.env.example`을 `agent/.env`로 복사한다.
`agent/.env`는 커밋하지 않는다.

```bash
cp agent/.env.example agent/.env
```

운영 추론 pipeline의 scoring backend는 `TRACEMIND_AGENT_SCORING_BACKEND`로 명시한다.
`classifier_head_logits` 같은 scorer를 묵시 기본값으로 붙이지 않는다.

## Main Features

| Feature | Purpose | Start here |
|---|---|---|
| Captured text | raw captured text ingest, generated view, debug analysis, training source projection | [src/features/captured_text/README.md](src/features/captured_text/README.md) |
| Inference | embedding, scorer backend, local interpretation, pipeline orchestration | [src/features/inference/README.md](src/features/inference/README.md) |
| Wellbeing | family extension summary, timeseries, space-web, child support, family access | [src/features/wellbeing/README.md](src/features/wellbeing/README.md) |
| Training runtime | current TrainingTask 실행, Query SSL/FSSL local update adapter, usage ledger | [src/features/training_runtime/README.md](src/features/training_runtime/README.md) |
| Federation | server round fetch/upload client와 local participant flow | `src/features/federation/` |
| Assets | shared adapter asset runtime, sync, local asset provider | `src/features/assets/` |
| Language | preprocessing, translation, backtranslation helper | `src/features/language/` |
| Typing segments | browser/desktop typing segment ingest use case | `src/features/typing_segments/` |

Feature module 목록과 migration 원칙은 [src/features/README.md](src/features/README.md)에
있다.

## API And Contracts

FastAPI shell은 [src/api/README.md](src/api/README.md)를 기준으로 읽는다.

Route module은 request/response payload 변환, HTTP status 매핑, feature service 호출
흐름만 보여준다. runtime 객체 생성과 기본 조립은 `agent/src/runtime/`가 맡는다.

Agent-local API/UI payload 계약은 [src/contracts/README.md](src/contracts/README.md)가
소유한다.

| Contract area | Owner |
|---|---|
| Captured text / typing segment | `agent/src/contracts/*captured*`, `typing_segment_contracts.py` |
| Family access / wellbeing / child support | `agent/src/contracts/*wellbeing*`, `family_access_contracts.py`, `child_support_contracts.py` |
| FL/model/training shared payload | `shared/src/contracts/` |

`apps/family_extension`은 agent contract를 소비해 타입을 생성하지만 계약 의미의 owner가
아니다.

## Runtime Composition

[src/runtime/README.md](src/runtime/README.md)가 agent process object graph를 설명한다.

| File | Responsibility |
|---|---|
| `src/runtime/composition.py` | repository, provider, feature service graph 조립 |
| `src/runtime/state.py` | `app.state`에 설치할 runtime object 목록과 설치 규칙 |
| `src/runtime/env.py` | agent API shell이 읽는 runtime 환경값 정규화 |

`runtime/`은 route handler, feature 알고리즘 의미, repository SQL을 소유하지 않는다.
실행 조립의 경계만 소유한다.

## State And Data Ownership

| State | Owner |
|---|---|
| Raw captured text and generated views | agent-local captured text storage |
| Analysis events and interpretation state | agent-local inference/wellbeing storage |
| Child-support raw messages and conversation context | agent-local child-support storage |
| Family PIN/session state | agent-local family access storage |
| Training usage ledger | agent-local training runtime storage |
| FL round lifecycle and aggregation state | `main_server` |
| Shared payload meaning | `shared/src/contracts` |
| Algorithm objective and update computation | `methods/` |

Agent feature끼리는 다른 feature의 storage 내부 구현을 직접 import하지 않는다. 필요한
값은 계약 payload나 public service boundary를 통해 받는다.

## Model And Scoring Entry Points

모델이나 scorer 관련 흐름을 읽을 때는 아래 순서가 짧다.

| Goal | Start here |
|---|---|
| 실험/스크립트 embedding preset | `conf/execution_context/embedding_adapter/*.yaml` |
| agent embedding adapter factory | `agent/src/infrastructure/model_adapters/embedding/factory.py` |
| backtranslation runtime | `agent/src/features/language/backtranslation_service.py` |
| 운영 추론 pipeline 조립 | `agent/src/features/inference/pipeline_service.py` |
| scorer backend / local interpretation | `agent/src/features/inference/` |

## Development Notes

- agent 내부 단위 검증은 `agent/tests`에 둔다.
- 서버와의 상호작용 검증은 root `tests/` integration 시나리오로 올린다.
- 새 method/algorithm 의미는 agent feature module로 끌어오지 않는다.
- local objective, pseudo-label algorithm, SSL hook, adaptation 계산 core는 `methods/`에 둔다.
- agent는 선택된 core를 실행 가능한 local task로 변환하는 port/adapter 역할만 맡는다.
