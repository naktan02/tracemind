# Federated Shard Policy

`methods/federated/shard_policy/`는 FL 실험에서 client별 item assignment를
결정하는 순수 shard policy 계산을 둔다.

## 책임

- bootstrap split 이후 남은 item을 client shard로 deterministic assignment
- `label_dominant`, `dirichlet_label_skew` policy 산술
- item label accessor를 받아 row shape와 분리된 assignment 결과 생성

## 제외

- dataset file IO
- simulation report/artifact 저장
- `scripts.labeled_query_rows` 같은 entrypoint-local row shape
- Hydra config loading

위 실행 glue는 `scripts/experiments/fl_ssl/federated_simulation/`에 남긴다.
