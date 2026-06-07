# Prototype Training Inputs

`methods/prototype/training_inputs/`는 prototype score 기반 pseudo-label 학습 예시의
view 변환 core를 둔다.

## 책임

- source row와 base embedding을 adapter-applied embedding으로 변환
- prototype score로 `AnalysisEvent`를 재구성
- weak/strong multiview에서 selection view와 update view를 분리
- stored score snapshot을 현재 prototype/adapter 기준으로 재점수화

## 제외

- raw text embedding 실행
- agent-local analysis event 저장소 접근
- `EmbeddedTrainingExample` agent DTO wrapping
- runtime backend registry wiring

위 runtime glue는 `agent/src/services/training/backends/inputs/`에 남긴다.
