# Text Classifier Adaptation

`methods/adaptation/text_classifier/`는 텍스트 encoder를 포함하는 classifier
adaptation core를 소유한다. 기존 `lora_classifier` 구조를 장기적으로 대체하되,
classifier-head 자체처럼 modality와 독립적인 primitive는
`methods/adaptation/classification/`이 소유한다.

## 책임

- `peft_encoder/` 기반 PEFT-adapted encoder + classifier head adaptation variant
- text-specific model composition, local training, update payload/materialization
- PEFT text encoder state를 generic aggregation core 입력/출력으로 바꾸는 projection

## 금지

- `methods.federated_ssl.fedmatch` 직접 import
- 새 내부 코드에서 legacy `methods.adaptation.lora_classifier` import
- classifier-head source-of-truth 소유
- FedAvg weighted-average algorithm 직접 구현
- LoRA/DoRA mechanism 구현 소유

FedMatch 같은 method semantics는 `methods/federated_ssl/<method>/`가 소유하고,
LoRA/DoRA 같은 PEFT mechanism은 `methods/adaptation/peft_adapters/`가 소유한다.
