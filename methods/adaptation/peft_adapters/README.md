# PEFT Adapters

`methods/adaptation/peft_adapters/`는 LoRA, DoRA, RS-LoRA 같은 PEFT mechanism
builder와 registry를 소유한다. text classifier task나 update payload 의미는 소유하지
않는다.

## 책임

- PEFT adapter builder protocol
- adapter mechanism registry
- LoRA/DoRA별 target module 해석과 builder

## 금지

- classifier label schema 해석
- classifier head 학습 loop
- text classifier update payload 조립
- FL SSL method semantics 소유

현재 `methods/adaptation/peft/`와 `methods/adaptation/lora/`는 migration source이며,
이 경로가 장기 목표 위치다.
