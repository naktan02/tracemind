# Experiment Web AGENTS

## 역할

`apps/experiment_web`는 TraceMind 개발자용 실험 workspace UI다.

이 앱의 목적은 아래에 한정한다.

1. 현재 catalog를 읽는다.
2. track/entrypoint/block selection을 조합한다.
3. compile preview와 compatibility/error를 보여준다.
4. future phase에서 local run/workspace 저장을 붙일 준비를 한다.

## 변경 규칙

- UI는 source of truth가 아니다.
- track separation을 흐리지 않는다.
  - `seed`
  - `central_adaptation`
  - `federated_runtime`
- metadata-only surface를 runnable처럼 보이게 만들지 않는다.
- compile error/warning은 숨기지 말고 그대로 드러낸다.
- dataset provenance/readiness는 설명용으로 보여주되, dataset import lane과 섞지 않는다.

## 구현 스타일

- `React + TypeScript + Vite`를 기준으로 유지한다.
- 앱 내부 상태는 최대한 명시적으로 두고, runtime 계약을 재발명하지 않는다.
- 복잡한 styling framework를 먼저 넣기보다 Phase 목적에 맞는 얇은 CSS를 우선한다.
- `src/types.ts`는 backend payload에서 생성되는 파일로 유지하고 수동 수정하지 않는다.
