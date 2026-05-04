# Prototype Methods

`methods/prototype/`는 prototype builder, assignment, update, scoring, evidence,
training input mechanism을 둔다.

Prototype pack contract와 serialization은 `shared`가 소유하고, prototype 분석
실험 runner나 sweep은 `scripts/` 또는 `research/analysis/`가 소유한다.

현재 활성 구현:

- `scoring/`: embedding-prototype similarity와 category score 집계 policy
- `evidence/`: prototype score나 score snapshot을 `PseudoLabelEvidence`로 정규화
- `training_inputs/`: prototype score 기반 single/multiview 학습 입력 view 계산
