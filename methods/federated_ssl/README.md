# Federated SSL Methods

`methods/federated_ssl/`는 SSL local update와 federated aggregation을 조합하는
method descriptor와 composition metadata를 둔다.

## 책임

- `federated_ssl_method` 이름별 descriptor
- client trainer, pseudo-labeler, view generator, server aggregator 조합 metadata
- 실제 runtime을 붙이기 전 method 후보의 구현 상태 표현

## 제외

- Hydra config loading
- simulation loop와 artifact/report 저장
- `LocalTrainingService`, `RoundOpenRequest` 같은 runtime 객체 생성

위 실행 glue는 `scripts/experiments/federated_simulation/`에 남긴다.
