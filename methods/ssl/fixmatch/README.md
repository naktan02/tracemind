# FixMatch

`methods/ssl/fixmatch/`는 USB 스타일 FixMatch objective core를 소유한다.

## 책임

- weak view probability에서 pseudo-label과 confidence mask 계산
- strong view consistency cross-entropy 계산
- labeled supervised loss와 unlabeled consistency loss 결합
- query SSL trainer가 호출하는 `FixMatchAlgorithm` adapter 제공

## 제외

- unlabeled row 준비와 augmentation cache 생성
- LoRA model/tokenizer 생성
- Hydra config loading
- run artifact 저장

위 실행 조립은 `scripts/experiments/lora_classifier/query_ssl/`와 해당
entrypoint가 담당한다.
