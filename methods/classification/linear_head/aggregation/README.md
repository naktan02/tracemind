# Linear Head Aggregation Projection

`methods/classification/linear_head/aggregation/`은 linear-head state/update를
generic aggregation core가 소비할 수 있는 입력으로 바꾸고, 결과를 다음 state로
materialize한다.

## 파일

- `linear_head_fedavg_projection.py`: classifier-head update payload를 FedAvg
  strategy 입력으로 변환하고 next `ClassifierHeadState`를 만든다.
