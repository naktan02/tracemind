# Family Extension

TraceMind 가족용 Chrome extension UI shell이다. 로컬 agent가 만든 wellbeing summary와
child-support payload를 소비해서 아이용 화면, 보호자 화면, popup entry, 입력 collector를
제공한다.

이 앱은 wellbeing 계산 owner가 아니다. 위험 추론, 카테고리 판단, 학습 buffer,
wellbeing projection은 agent가 소유하고, extension은 로컬 결과를 표시하고 captured text
event를 agent에 전달한다.

## What It Includes

- 최초 1회 child/parent PIN 설정 화면
- 확장 아이콘용 compact popup
- child route의 현재 상태 카드, AI 마음 도움, 위험도 변화 그래프, 공간웹 그래프
- parent route의 현재 상태 요약과 대응 방향 안내
- popup 기반 role별 PIN unlock과 session guard
- local agent 연결 상태 배너
- content script 기반 입력 surface collector
- agent가 꺼져 있을 때 재전송할 extension local queue
- collector debug page와 replay test fixture

부모 화면은 원문 텍스트, 카테고리 공간웹, 상세 추이 그래프를 직접 노출하지 않는다.

## Run In Development

먼저 agent API를 실행한다.

```bash
uv run uvicorn agent.src.api.main:app --reload --host 127.0.0.1 --port 8001
```

frontend 의존성을 설치하고 dev server를 연다.

```bash
cd apps/family_extension
npm install
npm run dev
```

기본 agent API target은 `http://127.0.0.1:8001`이다. 다른 주소를 쓰려면:

```bash
cd apps/family_extension
VITE_AGENT_API_BASE_URL=http://127.0.0.1:9001 npm run dev
```

기본 CORS 허용 origin은 아래 두 개다.

- `http://localhost:5174`
- `http://127.0.0.1:5174`

다른 origin을 열려면 backend에서 아래 env를 사용한다.

```bash
FAMILY_EXTENSION_ALLOWED_ORIGINS=http://localhost:5174,https://family.example.com \
uv run uvicorn agent.src.api.main:app --reload --host 127.0.0.1 --port 8001
```

## Build The Extension

```bash
cd apps/family_extension
npm run build
```

빌드 결과는 `apps/family_extension/dist/`에 생성된다. Chrome에서 개발용으로 확인할
때는 `dist`를 Load unpacked로 로드한다.

Chrome extension manifest source는 `public/manifest.json`이다.

## UI Surfaces

| Entry | Purpose |
|---|---|
| `index.html` | 본인용/가족용 React app entry. setup 전에는 `/setup`, 이후 `/child` 또는 `/parent` 흐름을 탄다 |
| `parent.html` | 부모용 상세 entry. 세션이 없으면 popup PIN 입력 안내를 보여준다 |
| `popup.html` | 확장 아이콘 compact popup. 본인/부모 페이지 진입과 debug 도구 진입을 제공한다 |
| `collector-debug.html` | 개발 중 마지막 captured segment JSON을 확인하는 debug page |
| `assets/content.js` | 웹페이지 입력 surface에 주입되는 content script |
| `assets/background.js` | segment queue와 local agent 전송을 맡는 background service worker |

`index.html`과 `parent.html`은 같은 React app shell인 `src/ui/main.tsx`를 사용한다.
`popup.html`은 `src/popup/popup.ts`를 사용한다.

## Source Layout

| Path | Responsibility |
|---|---|
| `src/ui/` | child/parent detail, route, React component, UI 전용 API client |
| `src/popup/` | extension icon popup 상태와 role 진입 |
| `src/common/` | UI/background/content script가 함께 쓰는 얇은 helper |
| `src/contracts/` | generated contract type |
| `src/collector/` | content script 입력 surface 감시와 segment 생성 |
| `src/extension/` | background service worker queue, local agent 전달, storage key |
| `src/dev/` | collector debug page script |
| `tests/collector/` | collector unit/replay test |

UI는 wellbeing/child-support 의미를 재정의하지 않고 agent API payload를 표시한다.
공간웹 그래프도 agent가 계산한 `nodes`/`edges` view-model만 렌더링한다.

## Collector Flow

collector는 raw key event stream을 저장하지 않는다. 안정화된 editor snapshot에서
사용자-visible text 후보를 만들고, background service worker가 agent-local captured
text event로 정규화해 전달한다.

```text
TypingSegmentPayload
-> CapturedTextEventPayload
-> agent-local CapturedTextRecord
-> CapturedTextGeneratedViewRecord
-> TrainingExampleSource
```

extension local storage queue는 agent가 꺼져 있을 때의 재전송 buffer일 뿐이며, 학습
데이터 source of truth가 아니다.

| Path | Role |
| --- | --- |
| `src/collector/content.ts` | 입력 surface 감지와 editor snapshot 읽기 |
| `src/collector/canonicalText.ts` | 사용자-visible text 기준 정규화 |
| `src/collector/hangulIme.ts` | 한글 IME 조합 상태 해석 |
| `src/collector/{segmentText,textDiff,segmentBuffer}.ts` | diff, idle flush, payload emit lifecycle |
| `src/extension/background.ts` | segment queue, local agent 전달, collector status |

## Collector Checks

collector unit/replay test:

```bash
cd apps/family_extension
npm run test:collector
```

확장 아이콘 popup에서 `debug 켜기`를 누르고 `debug 열기`로 debug page를 열면 마지막
`TypingSegmentPayload` JSON을 확인할 수 있다. segment는 5초 idle 후 생성된다.
agent에 inference pipeline이 아직 연결되지 않은 실행에서는 agent 응답이 503으로 남고,
수집 여부는 popup/debug page에서 확인한다.

## Contract Types

[src/contracts/generated.ts](src/contracts/generated.ts)는 generated file이다.
shared `wellbeing_signal` 또는 `family_access` contract를 바꾼 뒤에는 repo root에서
아래를 다시 실행한다.

```bash
./.venv/bin/python scripts/codegen/generate_family_extension_types.py
```

## Current Scope

현재 포함하지 않는 것:

- 장기 기록용 별도 부모 웹 대시보드
- 원격 agent 선택 UI
- 다중 자녀 프로필
- cloud LLM provider opt-in 대화
- persisted personalization state와의 정밀 연결
