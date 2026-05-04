# Prototype Methods

`methods/prototype/`는 prototype builder, assignment, update, evidence
mechanism을 둔다.

Prototype pack contract와 serialization은 `shared`가 소유하고, prototype 분석
실험 runner나 sweep은 `scripts/` 또는 `research/analysis/`가 소유한다.

현재 활성 구현:

- `evidence/`: prototype score나 score snapshot을 `PseudoLabelEvidence`로 정규화
