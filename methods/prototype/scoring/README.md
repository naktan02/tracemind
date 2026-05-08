# Prototype Scoring

`methods/prototype/scoring/`은 embedding과 prototype 사이의 score를 계산하고,
category 단위 score로 접는 순수 계산 core를 둔다.

## 책임

- embedding/prototype vector의 canonical coercion
- cosine pairwise score 계산
- 다중 prototype category score 집계 policy
- prototype score policy registry wiring

## 파일 역할

- `base.py`: score policy protocol
- `similarity.py`: vector coercion과 pairwise/category score 계산
- `policy_registry.py`: score policy lookup/decorator registry
- `score_policies/`: implementation-local score policy와 decorator registration

새 policy는 `score_policies/<policy>.py`에 두고 구현 옆 decorator로 등록한다.
runtime은 `policy_registry.py`의 이름 기반 builder를 사용하고, concrete class가
필요한 테스트/명시적 실험 코드는 `score_policies/<policy>.py`에서 직접 import한다.

## 제외

- agent-local query buffer 저장소 접근
- raw text retention
- classifier head shared state logits 계산
- inference/training runtime backend registry wiring

위 runtime glue는 `agent/src/services/inference/`와
`agent/src/services/training/`에 남긴다.
