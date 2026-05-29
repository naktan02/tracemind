# Prototype Evidence

`methods/prototype/evidence/`는 prototype score나 stored score snapshot을
공통 `PseudoLabelEvidence`로 정규화하는 순수 계산을 둔다.

## 책임

- category score ranking
- top1/top2/margin 기반 `PseudoLabelEvidence` 생성
- score distribution helper

## 제외

- agent-local query buffer 저장소 접근
- raw text retention
- training task routing과 backend registry wiring

위 runtime glue는 `agent/src/services/training/`에 남긴다.
