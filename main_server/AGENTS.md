# Main Server AGENTS

## 역할

`main_server/`는 round lifecycle, aggregation, artifact publication,
federated orchestration을 소유한다.

## 먼저 읽을 것

1. `main_server/README.md`
2. `main_server/src/services/README.md`
3. `main_server/src/services/federation/rounds/README.md`
4. 관련 shared contract와 domain entity

## 변경 규칙

- 서버는 개인 raw text나 개인 해석 상태를 소유하지 않는다.
- 서버 소유 책임은 round 상태, model revision, aggregation, publication으로
  제한한다.
- aggregation policy와 실행 mechanism을 분리한다.
- 새 backend나 acceptance policy를 추가할 때는 family/strategy 경계가
  코드에서 독립적으로 보이게 유지한다.
- 새 FL SSL method를 추가하기 위해 `main_server`에 method-specific round,
  aggregation, server policy 파일을 추가하지 않는다. method-local server/round
  policy 의미는 `methods/federated_ssl/<method>/`에 두고, `main_server`는
  lifecycle, validation, artifact materialization, publication adapter만 맡는다.
- aggregation method의 generic 산술/strategy wiring은
  `methods/federated/aggregation/`이 소유하고, adapter family별 delta 해석과
  next-state projection은 `methods/adaptation/<family>/`가 소유한다. server
  aggregation module은 selected methods strategy 호출 runtime adapter에 머문다.
- payload format 변경은 반드시 `shared` contract 변경과 함께 진행한다.

## 테스트 규칙

- 서버 내부 단위 검증은 `main_server/tests`에 둔다.
- multi-agent HTTP, end-to-end federation 검증은 root `tests/`에 둔다.
