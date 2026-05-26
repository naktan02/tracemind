# Feature Head Text Classifier

`feature_head/`는 고정 feature 위의 classifier head adaptation variant를 소유한다.
기존 `methods/adaptation/classifier_head/`를 장기적으로 옮길 목표 위치다.

## 책임

- classifier head training/evaluation/scoring primitive
- feature-head update payload 조립과 materialization
- feature-head state projection 입력 생성

## 금지

- PEFT adapter mechanism 구현
- FedAvg weighted-average algorithm 구현
- FedMatch/FedLGMatch 같은 FL SSL method semantics 소유

기존 `classifier_head/`는 migration 동안 compatibility shim으로만 남긴다.
