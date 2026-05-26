# Feature Head Text Classifier

`feature_head/`는 고정 feature 위의 classifier head adaptation variant를 소유한다.
기존 `methods/adaptation/classifier_head/` direct path는 migration 동안
compatibility shim으로만 남긴다.

## 책임

- classifier head training/evaluation/scoring primitive
- feature-head update payload 조립과 materialization
- feature-head state projection 입력 생성

## 파일

- `bootstrap.py`: prototype centroid 기반 classifier-head 초기 state 생성
- `scoring.py`: shared classifier-head state 기반 logits scoring backend

## 금지

- PEFT adapter mechanism 구현
- FedAvg weighted-average algorithm 구현
- FedMatch/FedLGMatch 같은 FL SSL method semantics 소유

FedAvg projection과 next-state materialization은 sibling package인
`methods/adaptation/text_classifier/aggregation/feature_head_fedavg_projection.py`가
소유한다.
