# Privacy Guards

`methods/adaptation/privacy_guards/`는 shared adapter update에 적용하는 privacy/safety
policy를 소유한다. 현재 구현은 no-op과 clip-only guard이며, 이후 DP noise 같은
업데이트 보호 정책이 추가될 수 있다.

## 책임

- privacy guard 이름, catalog metadata, registry
- adapter-kind별 update clipping/DP 계산
- `TrainingTask`의 privacy/clip 설정을 shared adapter update에 적용하는 policy

## 제외

- agent-local private state, query buffer, artifact upload
- model training loop와 objective loss 계산
- server aggregation policy와 artifact materialization
- SSL method identity, FedMatch/FedLGMatch 같은 논문 method 의미

agent/main_server는 선택된 guard를 실행 흐름에 연결만 한다. 새 guard가 round state,
secure aggregation protocol, server aggregation semantics를 요구하면 이 폴더 안에
억지로 넣지 말고 `shared` contract나 federated runtime capability를 함께 열어야 한다.
