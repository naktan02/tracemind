# Text Classifier Adaptation

`methods/adaptation/text_classifier/`는 텍스트 분류 task family의 adaptation
core를 소유한다. 이 경로는 기존 `lora_classifier` 구조를 장기적으로 대체하기 위한
목표 위치이며, 현재 단계에서는 code import를 아직 이동하지 않는다.

## 책임

- text classifier label/logit/metric/tensor state 공통 helper
- `feature_head/` 기반 classifier-head adaptation variant
- `peft_encoder/` 기반 PEFT-adapted encoder + classifier head adaptation variant
- text classifier state를 generic aggregation core 입력/출력으로 바꾸는 projection

## 금지

- `methods.federated_ssl.fedmatch` 직접 import
- 새 내부 코드에서 legacy `methods.adaptation.lora_classifier` import
- FedAvg weighted-average algorithm 직접 구현
- LoRA/DoRA mechanism 구현 소유

FedMatch 같은 method semantics는 `methods/federated_ssl/<method>/`가 소유하고,
LoRA/DoRA 같은 PEFT mechanism은 `methods/adaptation/peft_adapters/`가 소유한다.
