# Apps AGENTS

## 역할

`apps/`는 개발자용 실험 UI와 future 제품 UI shell을 두는 계층이다.

중요:

- `apps`는 source of truth가 아니다.
- 계약 의미, 전략 이름, 실행 기본값은 `shared`, `main_server`, `scripts/conf`가 소유한다.
- UI는 catalog/API/contract를 소비하고 보여주는 역할만 맡는다.

## 변경 규칙

- UI가 contract 의미를 다시 정의하지 않는다.
- backend payload shape를 프런트 전용 임의 포맷으로 조용히 바꾸지 않는다.
- lane 분리는 화면에서도 유지한다.
  - `experiment_web`
  - future `youth_*`
  - future `guardian_*`
- developer tool과 product UI의 문구/권한/정보 노출을 섞지 않는다.
- 저장이 필요해도 대용량 artifact를 앱 로컬 저장소의 source of truth로 두지 않는다.

## 구현 원칙

- 기본은 `TypeScript` 기반으로 간다.
- 앱 shell은 얇게 유지하고, 공통 타입/클라이언트는 재사용 가능한 위치로 분리할 수 있게 만든다.
- `experiment_web`는 compile/run consumer이지 Hydra/script contract owner가 아니다.
- future youth/guardian UI는 output/view-model consumer이지 inference/training owner가 아니다.
