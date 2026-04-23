# Family Extension

`apps/family_extension`는 TraceMind 가족용 확장 프로그램 MVP shell이다.

현재 단계는 `Phase 7` 범위에 해당한다.

포함하는 것:

1. popup용 child summary entry
2. parent detail용 별도 entry
3. `/child`, `/unlock`, `/parent` route shell
4. 로컬 프로그램 wellbeing API를 호출할 준비가 된 client layer
5. child view의 현재 상태 카드, 요약 문구, 행동 제안
6. 연결 상태 배지와 기본 navigation
7. generated contract type file
8. parent PIN unlock API 연결
9. 실패 횟수/잠금 상태 안내
10. 세션 기반 parent route guard
11. parent summary 카드
12. 7d / 14d / 30d 추이 그래프
13. 부모용 권장 행동/세션 정보 카드

아직 포함하지 않는 것:

1. low-data/offline UX 고도화
2. 실제 wellbeing 엔진 결과 적재
3. 장기 기록용 별도 부모 웹 대시보드

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
  - 기본 route는 `/child`
- `parent.html`
  - 부모용 상세 entry
  - 기본 route는 `/parent`

Chrome extension manifest는 `public/manifest.json`을 source로 사용한다.

## 타입 동기화

- [src/contracts/generated.ts](/home/jmgjmg102/tracemind_server/apps/family_extension/src/contracts/generated.ts)는 generated file이다.
- shared wellbeing contract를 바꾼 뒤에는 repo root에서 `./.venv/bin/python scripts/generate_family_extension_types.py`를 다시 실행한다.
