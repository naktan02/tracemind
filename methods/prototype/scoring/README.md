# Prototype Scoring

`methods/prototype/scoring/`은 embedding과 prototype 사이의 score를 계산하고,
category 단위 score로 접는 순수 계산 core를 둔다.

## 책임

- embedding/prototype vector의 canonical coercion
- cosine pairwise score 계산
- 다중 prototype category score 집계 policy
- prototype score policy registry wiring

## 제외

- agent-local query buffer 저장소 접근
- raw text retention
- classifier head shared state logits 계산
- inference/training runtime backend registry wiring

위 runtime glue는 `agent/src/services/inference/`와
`agent/src/services/training/`에 남긴다.
