# Inference Feature

`inference/`는 agent-local 입력을 analysis event와 개인 기준 해석으로 바꾸는
로컬 추론 feature다. 외부 model adapter mechanism은
`agent/src/infrastructure/model_adapters/`가 소유하고, 이 feature는 adapter를 호출해
pipeline과 해석 흐름을 조립한다.

읽기 시작점:

- `pipeline_service.py`: preprocess 이후 embedding, scoring, interpretation을 한 번의
  로컬 추론 흐름으로 묶는다.
- `pipeline_factory.py`: agent runtime에서 기본 pipeline service를 조립한다.
- `scoring_service.py`: scorer backend를 호출해 category score를 계산한다.
- `scoring_backends/`: 교체 가능한 scorer backend agent runtime adapter.
- `interpretation/`: raw score를 baseline/time-series/decision policy로 해석한다.

새 scorer 계산 core가 필요하면 먼저 `methods/`에 두고, 이 feature에는 agent runtime
adapter만 둔다.
