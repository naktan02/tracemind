# Inference Interpretation

이 package는 scorer backend가 만든 category score를 agent-local 개인 기준으로 해석한다.
raw score 계산은 `agent/src/services/inference/scoring_service.py`와
`scoring_backends/`가 소유하고, 이 package는 baseline, time-series 누적, 최종 decision
policy와 결과 value object를 소유한다.

읽기 시작점:

- `baseline.py`: 과거 analysis event 기반 개인 baseline profile 계산.
- `time_series.py`: category score의 delta, EWMA, persistence 누적.
- `decision.py`: baseline/time-series/policy를 조합하는 해석 use case.
- `decision_policy.py`: agent-local rule 기반 최종 판단 정책.
- `state.py`: baseline과 time-series 상태 value object.
- `result.py`: 최종 assessment result value object.
