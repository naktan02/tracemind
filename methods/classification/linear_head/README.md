# Linear Head Classification

`methods/classification/linear_head/`는 고정 feature 또는 embedding 위에 얹는
modality-independent linear classifier head primitive를 소유한다.

## 책임

- 고정 feature 위 classifier-head state 생성과 scoring
- classifier-head update를 generic aggregation core 입력으로 바꾸는 projection
- classification task에서 공유되는 payload-family primitive
- `payload_adapter_module.py`에서 shared `classifier_head` contract alias를 이
  implementation root에 연결

## 제외

- text encoder, tokenizer, PEFT text model composition
- LoRA/DoRA 같은 PEFT mechanism 구현
- FedAvg weighted-average algorithm 자체

텍스트 encoder에 종속되는 구현은 `methods/adaptation/peft_text_encoder/`에
둔다.

`aggregation/linear_head_fedavg_projection.py`는 classifier-head update payload를
FedAvg strategy 입력으로 변환하고 다음 `ClassifierHeadState`를 materialize한다.
FedAvg weighted-average 산술 자체는 `methods/federated/aggregation/`이 소유한다.
