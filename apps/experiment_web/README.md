# Experiment Web

`apps/experiment_web`는 TraceMind 개발자용 experiment workspace UI다.

현재 Phase 4 범위:

1. experiment catalog 읽기
2. lane page 단위 step workflow
   - 실행 작업 선택
   - preset block별 조합
   - 검토 및 실행
3. typed override field + advanced JSON patch 입력
4. compile preview / error / warning 표시
5. workspace 저장 / 재열기
6. local run launch
7. recent run status / stdout / stderr / artifact path 표시
8. 기존에 생성된 split/checkpoint artifact를 source picker에 자동 노출
9. 상단 saved workspace compare board
   - multi-select metric columns
   - 같은 기본 조합끼리 결과 비교
   - `불러와 수정`
   - `복제 후 수정`
   - `그대로 재실행`
   - `삭제`

중요:

- 이 앱은 Hydra 파일 본문을 직접 수정하지 않는다.
- catalog가 노출한 typed field와 run-local override patch만 덧씌워 preview/run surface를 만든다.
- [src/types.ts](/home/jmgjmg102/tracemind_server/apps/experiment_web/src/types.ts)는 generated file이다.
- backend payload/contract를 바꾼 뒤에는 repo root에서 `./.venv/bin/python scripts/codegen/generate_experiment_web_types.py`를 다시 실행한다.
- 사용 순서는 `탭 선택 -> 저장된 실험 비교 -> 새 조합 편집 -> 저장/실행`으로 본다.
- 중앙 적응의 `적응 데이터 소스`와 `초기 체크포인트`는 정적 Hydra preset뿐 아니라
  `data/processed/**` 아래 기존 생성 artifact도 선택지로 보여준다.

아직 포함하지 않는 것:

1. FL runtime 실행 전체
2. hybrid multi-component composition
3. multi-user/shared queue

## 개발 실행

1. backend API 실행

```bash
uvicorn main_server.src.api.main:app --reload
```

2. frontend 의존성 설치

```bash
cd apps/experiment_web
npm install
```

3. dev server 실행

```bash
cd apps/experiment_web
npm run dev
```

기본 API target은 `http://127.0.0.1:8000`이다.

다른 API 주소를 쓰려면:

```bash
cd apps/experiment_web
VITE_API_BASE_URL=http://127.0.0.1:9000 npm run dev
```

기본 CORS 허용 origin은 아래 두 개다.

- `http://localhost:5173`
- `http://127.0.0.1:5173`

다른 origin을 열려면 backend에서 아래 env를 사용한다.

```bash
EXPERIMENT_WEB_ALLOWED_ORIGINS=http://localhost:5173,https://experiment.example.com \
uvicorn main_server.src.api.main:app --reload
```
