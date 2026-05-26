# Legacy Text Classifier Feature Head

`methods/adaptation/text_classifier/feature_head/`는 이전 migration 단계에서 만든
호환 경로다. 새 source of truth는
`methods/adaptation/classification/feature_head/`이다.

## 책임

- 기존 `bootstrap.py`, `scoring.py` direct path를 named-symbol compatibility shim으로 유지

## 금지

- business rule, source-of-truth 상수, registry metadata 소유
- PEFT adapter mechanism 구현
- FedAvg weighted-average algorithm 구현
- FedMatch/FedLGMatch 같은 FL SSL method semantics 소유

FedAvg projection과 next-state materialization의 source of truth는
`methods/adaptation/classification/aggregation/feature_head_fedavg_projection.py`다.
