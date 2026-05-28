# PEFT Adapters

`methods/adaptation/peft_adapters/`는 LoRA, DoRA, RS-LoRA 같은 PEFT mechanism
builder와 registry를 소유한다. text classifier task나 update payload 의미는 소유하지
않는다.

## 책임

- PEFT adapter builder protocol
- adapter mechanism registry와 `methods/adaptation/peft_adapters/<mechanism>/builder.py`
  convention import
- LoRA/DoRA별 target module 해석과 builder

## 금지

- classifier label schema 해석
- classifier head 학습 loop
- text classifier update payload 조립
- FL SSL method semantics 소유
- registry 파일에 concrete mechanism import 목록 누적

새 mechanism은 실제 실행 구현이 필요할 때만
`methods/adaptation/peft_adapters/<mechanism>/builder.py`와
`conf/strategy_axes/model_architecture/peft/<mechanism>.yaml`로 연다. `lora`,
`dora` 같은 mechanism 이름을 `trainable_state/update_family` leaf나 `scripts`
분기로 만들지 않는다.

기존 `methods/adaptation/peft/`와 `methods/adaptation/lora/` compatibility
package는 제거했다. 새 내부 코드는 이 경로를 직접 import한다.
