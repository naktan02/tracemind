# Experiment Dashboard AGENTS

## 역할

`apps/experiment_dashboard`는 실험 결과 index를 읽는 research/developer UI다.

- 실험 metric 의미와 schema source of truth를 소유하지 않는다.
- `scripts/workflows/result_index`가 export한 정적 JSON bundle을 소비한다.
- product UI, family extension, agent/main_server runtime과 섞지 않는다.

## 규칙

- dataset/file path를 화면 계약으로 만들지 않는다. `ourafla_reddit` 같은 semantic
  name과 `run_id`만 표시한다.
- 앱 로컬 저장소를 source of truth로 두지 않는다.
- 서버 API가 필요해지기 전까지 정적 JSON consumer로 유지한다.
