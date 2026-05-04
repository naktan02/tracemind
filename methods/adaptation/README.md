# Adaptation Methods

`methods/adaptation/`은 학습 adapter 적용 core를 둔다.

현재 활성 구현:

- `peft/`: PEFT adapter builder protocol과 registry
- `lora/`: LoRA/RSLoRA builder core

rank, alpha, target module 같은 실행 파라미터는 code folder가 아니라 config에서
선택한다. DoRA, IA3, linear probe, full fine-tune은 실제 구현이 필요해질 때
별도 method로 추가한다.
