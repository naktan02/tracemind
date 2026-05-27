# Linear Head Classification

`methods/classification/linear_head/`는 고정 feature 또는 embedding 위에 얹는
modality-independent linear classifier head primitive를 소유한다.

## 책임

- 고정 feature 위 classifier-head state 생성과 scoring
- classifier-head update를 generic aggregation core 입력으로 바꾸는 projection
- classification task에서 공유되는 adapter-family primitive

## 제외

- text encoder, tokenizer, PEFT text model composition
- LoRA/DoRA 같은 PEFT mechanism 구현
- FedAvg weighted-average algorithm 자체

텍스트 encoder에 종속되는 구현은 `methods/adaptation/peft_text_classifier/`에
둔다.
