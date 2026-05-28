# PEFT Text Classifier

`methods/adaptation/peft_text_classifier/`는 PEFT-adapted text encoder와 task
classifier head를 함께 학습하는 update family를 소유한다. PEFT mechanism 자체는
별도 소유자가 있으며, 이 package는 text classifier task payload, training, update
materialization, FL SSL execution primitive를 맡는다.

## 책임

- PEFT adapter가 적용된 text encoder와 classifier head composition
- supervised/query SSL local training loop와 delta extraction
- local update payload builder와 artifact materialization
- FL SSL adapter-family execution primitive 연결
- `initial_query_ssl_algorithm_state` 입력과 `query_ssl_algorithm_state` 결과 보존
- `adapter_family_module.py`에서 active `peft_classifier` contract를 이
  implementation root에 연결

## 금지

- LoRA/DoRA mechanism builder 자체 소유
- FedMatch partition 이름, original parameter, objective 의미 직접 소유
- server/main runtime orchestration 소유

LoRA/DoRA mechanism 구현은 `methods/adaptation/peft_adapters/`에 두고, FedMatch
method semantics는 `methods/federated_ssl/fedmatch/`에서 callable/config로 주입한다.
