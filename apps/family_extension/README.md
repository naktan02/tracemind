# Family Extension

`apps/family_extension`는 TraceMind 가족용 확장 프로그램 MVP shell이다.

현재 단계는 `Phase 10` 범위에 해당한다.

## 현재 포함하는 것

1. popup용 기본 entry
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
  - popup entry
  - 초기 setup 전에는 `/setup`
  - setup 완료 후 잠금 상태면 `/gate`
- `parent.html`
  - 부모용 상세 entry
  - 세션이 없으면 `/gate` 또는 `/parent/unlock` 흐름으로 정규화된다

Chrome extension manifest는 `public/manifest.json`을 source로 사용한다.

## 타입 동기화

- [src/contracts/generated.ts](src/contracts/generated.ts)는 generated file이다.
- shared `wellbeing_signal` 또는 `family_access` contract를 바꾼 뒤에는 repo root에서 아래를 다시 실행한다.

```bash
./.venv/bin/python scripts/codegen/generate_family_extension_types.py
```
