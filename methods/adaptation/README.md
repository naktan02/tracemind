# Adaptation Methods

`methods/adaptation/`은 학습 adapter 적용 core를 둔다.

## 하위 패키지 지도

- `diagonal_scale/`: diagonal-scale heuristic local update 계산 core
- `peft/`: PEFT adapter builder protocol과 registry
- `lora/`: LoRA/RSLoRA builder core
- `lora_classifier/`: frozen backbone + LoRA/PEFT adapter + classifier head
  재사용 scaffold
- `query_classifier_adaptation/`: query-domain LoRA/classifier 중앙 실험의
  token-batch 입력 glue와 기존 import compatibility shim

rank, alpha, target module 같은 실행 파라미터는 code folder가 아니라 config에서
선택한다.

## 추가 기준

DoRA, IA3, linear probe, full fine-tune은 실제 실험/런타임 경계가 필요해질 때
별도 method로 추가한다. 이름만 미리 열어 두지 않는다.
