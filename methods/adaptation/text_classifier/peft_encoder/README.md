# PEFT Encoder Text Classifier

`peft_encoder/`는 순수 encoder만 뜻하지 않는다. 이 경로는
“PEFT-adapted encoder + task classifier head” text classifier adaptation variant를
소유한다.

## 책임

- PEFT adapter가 적용된 text encoder와 classifier head composition
- supervised/query SSL local training loop와 delta extraction
- local update payload builder와 artifact materialization
- FL SSL adapter-family execution primitive 연결
- `initial_query_ssl_algorithm_state` 입력과 `query_ssl_algorithm_state` 결과 보존

## 금지

- LoRA/DoRA mechanism builder 자체 소유
- FedMatch partition 이름, original parameter, objective 의미 직접 소유
- server/main runtime orchestration 소유

LoRA/DoRA mechanism 구현은 `methods/adaptation/peft_adapters/`에 두고, FedMatch
method semantics는 `methods/federated_ssl/fedmatch/`에서 callable/config로 주입한다.
