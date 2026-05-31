# PEFT Text Encoder Federated SSL Primitives

`peft_encoder/federated_ssl/`은 PEFT text encoder variant가 FL SSL method와 만나는
update-family 실행 primitive를 소유한다.

## 책임

- method-owned local training bridge
- helper provider와 peer prediction snapshot materialization
- supervised seed step primitive
- server update policy를 PEFT text encoder aggregation projection으로 해석
- partitioned trainable state runtime primitive

## 금지

- 논문 method semantics 소유
- method 이름이 들어간 파일 증식
- `methods.federated_ssl.<method>` 직접 import

method package는 partition plan, objective callable, helper policy를 제공하고, 이
경로는 전달받은 primitive를 실행한다.
