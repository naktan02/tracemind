# Agent Features

`features/`는 agent를 일반 백엔드형 feature module 구조로 옮기는 경계다.
아직 이동하지 않은 기능은 기존 `services/`, `api/`, `runtime/`,
`infrastructure/repositories/`가 운영 source of truth이며, 이동이 끝난 기능은 이
디렉터리 아래가 source of truth다.

현재 이동된 feature:

- `captured_text/`: raw text ingest, generated view, debug analysis job, training
  source projection, captured text 전용 SQLite storage.
- `wellbeing/`: family extension wellbeing output, family access, child support,
  wellbeing 전용 SQLite storage.

## 소유 원칙

- feature module은 route, use case/service, feature 전용 repository/storage,
  payload projection을 한 읽기 경로에서 확인할 수 있게 둔다.
- SQLite connection, embedding/translation model adapter, HTTP transport처럼 여러
  feature가 공유하는 mechanism은 `infrastructure/`에 남긴다.
- agent process object graph와 environment wiring은 `runtime/`에 남긴다.
- agent-local API/UI payload 계약은 `agent/src/contracts/`가 계속 소유한다.
- feature끼리는 다른 feature의 storage 내부 구현을 직접 import하지 않는다. 필요한
  값은 계약 payload나 public service boundary를 통해 받는다.
- 새 method/algorithm 의미는 feature module로 끌어오지 않는다. local runtime adapter는
  선택된 `methods/` core를 실행 가능한 agent 작업으로 변환하는 역할만 맡는다.
- compatibility module을 만들지 않는다. 불가피한 경우 같은 phase에서 제거 조건을
  문서화하고 architecture guard 후보에 올린다.

## Migration 순서

1. feature module 기준선과 검증 gate를 고정한다.
2. `captured_text`를 pilot으로 옮기며 raw text lifecycle과 저장소 경계를 함께 검증한다.
3. `wellbeing`을 옮기며 signal, space-web, family access, child support를 feature 내부
   하위 Module로 유지한다.
4. `inference`를 옮기며 pipeline, scoring, interpretation 경계를 유지한다.
5. `training_runtime`을 옮기며 local update 실행과 method request build를 분리한다.
6. `federation`, `assets`, `language`, `typing_segments`처럼 cross-runtime feature를 옮긴다.
7. FastAPI router와 dependency wiring을 feature route 기준으로 정리한다.
8. 남은 `services/`/repository compatibility 표면을 제거하고 architecture guard를 고정한다.

각 phase는 하나의 concern만 이동한다. 같은 의미의 로직을 새 위치와 옛 위치에 복사하지
않고, phase가 끝날 때마다 `ruff`, `pytest`, import guard로 닫는다.
