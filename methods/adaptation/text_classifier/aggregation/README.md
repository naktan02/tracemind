# Text Classifier Aggregation Projection

`text_classifier/aggregation/`은 text classifier family state를 generic aggregation
core가 소비할 수 있는 입력으로 바꾸고, aggregation 결과를 family state로 되돌리는
projection 계층이다.

## 책임

- feature-head state projection
- PEFT encoder + classifier head state projection
- partitioned state projection과 materialization

## 금지

- weighted average policy 직접 구현
- FedAvg algorithm 재구현
- client weighting, server aggregation policy, round lifecycle 소유

FedAvg 산술과 strategy wiring은 `methods/federated/aggregation/`이 소유한다.
