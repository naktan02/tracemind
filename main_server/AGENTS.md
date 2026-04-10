# Main Server AGENTS

## 역할

`main_server/`는 round lifecycle, aggregation, artifact publication,
federated orchestration을 소유한다.

## 먼저 읽을 것

1. `main_server/README.md`
2. `main_server/src/services/README.md`
3. `main_server/src/services/rounds/README.md`
4. 관련 shared contract와 domain entity

## 변경 규칙

- 서버는 개인 raw text나 개인 해석 상태를 소유하지 않는다.
- 서버 소유 책임은 round 상태, model revision, aggregation, publication으로
  제한한다.
- aggregation policy와 실행 mechanism을 분리한다.
- 새 backend나 acceptance policy를 추가할 때는 family/strategy 경계가
  코드에서 독립적으로 보이게 유지한다.
- payload format 변경은 반드시 `shared` contract 변경과 함께 진행한다.

## 테스트 규칙

- 서버 내부 단위 검증은 `main_server/tests`에 둔다.
- multi-agent HTTP, end-to-end federation 검증은 root `tests/`에 둔다.
