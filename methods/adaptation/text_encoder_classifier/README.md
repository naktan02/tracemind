# Text Encoder Classifier

`methods/adaptation/text_encoder_classifier/`는 text encoder 위 linear head
supervised 학습에서 PEFT와 full fine-tuning이 공유하는 작은 scaffold를 소유한다.

## 책임

- text encoder pooled representation + linear head model wrapper
- trainable surface manifest 정규화
- supervised CE 학습/evaluation loop
- transformer tokenizer/backbone loading helper
- pooled representation 수집과 2D reducer 계산 primitive

## 제외

- PEFT adapter mechanism 선택과 PEFT state/update payload
- full-model checkpoint artifact 저장 방식
- Hydra entrypoint orchestration
- projection artifact 저장과 논문용 figure 생성

PEFT-specific state와 FL update family 의미는 계속
`methods/adaptation/peft_text_encoder/`가 소유한다.
