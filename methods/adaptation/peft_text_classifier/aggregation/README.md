# PEFT Text Classifier Aggregation Projection

`peft_text_classifier/aggregation/`은 PEFT text classifier family state를 generic
aggregation core가 소비할 수 있는 입력으로 바꾸고, aggregation 결과를 family
state로 되돌리는 projection 계층이다.

## 책임

- PEFT encoder + classifier head state projection
- partitioned state projection과 materialization

## 파일

- `peft_encoder_fedavg_projection.py`: LoRA/PEFT encoder update payload를 generic
  FedAvg strategy 입력으로 바꾸고 next state artifact를 materialize
- `peft_encoder_partitioned_projection.py`: partitioned client delta를 병합/평균하고
  partitioned next state artifact를 materialize
- `peft_encoder_state_projection.py`: base global snapshot과 aggregated delta를 다음
  state/artifact payload로 투영
- `peft_encoder_partitioned_state.py`: partition별 delta merge와 residual split helper

## 금지

- weighted average policy 직접 구현
- FedAvg algorithm 재구현
- client weighting, server aggregation policy, round lifecycle 소유

FedAvg 산술과 strategy wiring은 `methods/federated/aggregation/`이 소유한다.
