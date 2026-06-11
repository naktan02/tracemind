# Full Text Encoder

`methods/adaptation/full_text_encoder/`는 중앙 실험용 full text encoder supervised
fine-tuning scaffold를 소유한다.

## 책임

- pretrained text encoder 전체와 linear head를 supervised CE로 학습할 모델 조립
- full-model trainable surface manifest와 parameter count 기록
- 중앙 실험 runner가 저장할 full encoder checkpoint provenance 제공

## 제외

- FL update payload, aggregation, publication runtime
- PEFT adapter mechanism
- agent/main_server runtime translation

현재 이 package는 중앙 실험용 비교 surface이며, production/FL update family로
승격하지 않는다.
