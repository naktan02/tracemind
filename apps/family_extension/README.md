# Family Extension

`apps/family_extension`는 TraceMind 가족용 확장 프로그램 MVP shell이다.

현재 단계는 `Phase 10` 범위에 해당한다.

## 현재 포함하는 것

1. 확장 아이콘용 compact popup entry
2. parent detail용 별도 entry
3. `/setup`, `/gate`, `/child/unlock`, `/parent/unlock`, `/child`, `/parent` route shell
4. `family_access`와 `wellbeing_signal` contract를 같이 소비하는 client layer
5. 최초 1회 child/parent PIN 설정 화면
6. role 선택 gate 화면
7. child role용 현재 상태 카드
8. parent role용 현재 상태 카드와 7d / 14d / 30d 추이 그래프
9. role별 PIN unlock API 연결
10. 실패 횟수/잠금 상태 안내
11. role session 기반 route guard
12. health polling 기반 연결 상태 배너
13. low-data / stale-data 안내 배너
14. scored event -> wellbeing snapshot projection을 통한 실제 로컬 출력 연결
15. role 화면 이탈 시 즉시 재잠금

## 내부 구조

`apps/family_extension`는 하나의 Chrome extension 패키지로 유지한다. UI와
향후 입력 수집 runtime은 같은 manifest에 묶이지만, 코드 책임은 분리한다.

- `src/ui/`
  - child/parent detail, route, React component, UI 전용 API client를 둔다.
  - wellbeing/child-support 의미를 재정의하지 않고 agent API payload를 표시한다.
- `src/popup/`
  - 확장 아이콘을 눌렀을 때 뜨는 compact 상태 popup을 둔다.
  - collector status, debug toggle, debug page 진입만 보여준다.
- `src/common/`
  - UI, background service worker, content script가 함께 쓸 수 있는 얇은 공통
    helper를 둔다.
  - 현재는 local agent API 호출 helper를 소유한다.
- `src/contracts/`
  - generated contract type만 둔다.
  - shared contract 변경 뒤 codegen으로 갱신한다.

`src/collector/`에는 content script의 입력 surface 감시와 segment 생성만 두고,
`src/extension/`에는 background service worker의 queue, local agent 전달, extension
storage key를 둔다. 위험 추론, 카테고리 판단, 학습 buffer는 계속 agent가 소유한다.

현재 collector runtime:

- `src/collector/content.ts`
  - `input`, `textarea`, `contenteditable`, rich editor surface를 감지한다.
  - IME composition commit 직후에는 짧은 settle window 뒤에 editor snapshot을 읽는다.
  - raw key event stream을 수집하지 않는다.
- `src/collector/canonicalText.ts`
  - editor snapshot을 사용자-visible text 기준 canonical text로 정규화한다.
  - rich editor가 조합 placeholder 뒤에 붙이는 zero-width sentinel은 여기에서 제거한다.
- `src/collector/hangulIme.ts`
  - 한글 IME 조합 중/확정/phantom delete 해석을 소유한다.
  - `ㄲ -> 깡`, `ㄴ -> 네`, 복합 모음/받침처럼 snapshot이 늦게 안정화되는 케이스를
    segment buffer 밖에서 정리한다.
- `src/collector/segmentText.ts`, `src/collector/textDiff.ts`
  - 안정화된 snapshot과 baseline의 diff로 final text 후보를 고른다.
  - 이벤트 data 누적은 final text의 source of truth로 쓰지 않는다.
- `src/collector/segmentBuffer.ts`
  - element별 baseline, idle flush, 삭제 기록, payload emit lifecycle만 소유한다.
- `src/extension/background.ts`
  - content script가 보낸 segment를 `chrome.storage.local` queue에 저장한다.
  - `http://127.0.0.1:8001/api/v1/typing-segments`로 local agent에 전송한다.
  - agent가 꺼져 있으면 queue를 유지하고 collector status에 오류를 남긴다.
- `collector-debug.html`
  - 개발용 extension page다.
  - debug 저장을 켠 경우 마지막 `TypingSegmentPayload` JSON을 보여준다.
  - raw segment를 보여주는 화면이므로 배포용 UI와 섞지 않는다.
- `tests/collector/replay-fixtures/`
  - 실제 입력 이벤트 순서를 JSON으로 보존하는 replay fixture다.
  - 브라우저 자동화 없이 `SegmentBuffer`에 이벤트를 재생해 한글 IME 회귀를 검증한다.

## 아직 포함하지 않는 것

1. 장기 기록용 별도 부모 웹 대시보드
2. 원격 agent 선택 UI
3. 다중 자녀 프로필
4. cloud LLM provider opt-in 대화
5. persisted personalization state와의 정밀 연결

## 개발 실행

1. agent API 실행

```bash
uvicorn agent.src.api.main:app --reload --port 8001
```

2. frontend 의존성 설치

```bash
cd apps/family_extension
npm install
```

3. dev server 실행

```bash
cd apps/family_extension
npm run dev
```

기본 agent API target은 `http://127.0.0.1:8001`이다.

다른 주소를 쓰려면:

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
uvicorn agent.src.api.main:app --reload --port 8001
```

## 확장 entry

- `index.html`
  - 아이용/가족용 React app entry
  - 초기 setup 전에는 `/setup`
  - setup 완료 후 잠금 상태면 `/gate`
- `popup.html`
  - 확장 아이콘 compact popup entry
  - collector status와 debug 진입을 제공한다
- `parent.html`
  - 부모용 상세 entry
  - 세션이 없으면 `/gate` 또는 `/parent/unlock` 흐름으로 정규화된다
- `assets/content.js`
  - 웹페이지 입력 surface에 주입되는 content script
- `assets/background.js`
  - segment queue와 local agent 전송을 맡는 background service worker
- `collector-debug.html`
  - 개발 중 마지막 segment JSON을 확인하는 debug page

`index.html`과 `parent.html`은 `src/ui/main.tsx`를 사용하고, `popup.html`은
`src/popup/popup.ts`를 사용한다.

## Collector 개발 점검

1. agent API를 실행한다.

```bash
uv run uvicorn agent.src.api.main:app --reload --host 127.0.0.1 --port 8001
```

2. 확장을 빌드하고 `dist`를 Load unpacked로 로드한다.

```bash
cd apps/family_extension
npm run build
```

3. collector unit/replay test를 실행한다.

```bash
cd apps/family_extension
npm run test:collector
```

4. 확장 아이콘 popup에서 `debug 켜기`를 누르고 `debug 열기`로 debug page를 연다.

debug 저장을 켠 뒤 fixture page에서 다시 입력하면 마지막 segment JSON을 확인할 수
있다. segment는 5초 idle 후 생성된다. agent에 inference pipeline이 아직 연결되지
않은 실행에서는 agent 응답이 503으로 남고, 수집 여부는 popup/debug page에서 확인한다.

Chrome extension manifest는 `public/manifest.json`을 source로 사용한다.

## 타입 동기화

- [src/contracts/generated.ts](src/contracts/generated.ts)는 generated file이다.
- shared `wellbeing_signal` 또는 `family_access` contract를 바꾼 뒤에는 repo root에서 아래를 다시 실행한다.

```bash
./.venv/bin/python scripts/codegen/generate_family_extension_types.py
```
