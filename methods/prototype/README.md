# Prototype Methods

`methods/prototype/`는 prototype scoring, evidence, training input처럼
prototype pack을 소비하는 method mechanism을 둔다.

Prototype pack contract, serialization, 공용 build strategy 표면은 `shared`가
소유한다. 현재 single/kmeans/dbscan builder는
`shared/src/services/prototypes/build_strategies.py`에 함께 둔다. 별도
`methods/prototype/building` 계층은 만들지 않는다.

Prototype 분석 실험 runner나 sweep은 `scripts/` 또는 `research/analysis/`가
소유한다.

현재 활성 구현:

- `scoring/`: embedding-prototype similarity와 category score 집계 policy
- `evidence/`: prototype score나 score snapshot을 `PseudoLabelEvidence`로 정규화
- `training_inputs/`: prototype score 기반 single/multiview 학습 입력 view 계산
