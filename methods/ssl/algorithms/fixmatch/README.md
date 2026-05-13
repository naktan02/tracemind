# FixMatch

`methods/ssl/algorithms/fixmatch/`는 USB 스타일 FixMatch objective core를
소유한다.

## 책임

- weak view probability에서 pseudo-label과 confidence mask 계산
- strong view consistency cross-entropy 계산
- labeled supervised loss와 unlabeled consistency loss 결합
- query SSL trainer가 호출하는 `FixMatchAlgorithm` adapter 제공

입력 경계:

- 저장 row surface는 strict USB형 `text + aug_0 + aug_1`이다.
- dataloader가 `text`를 weak view로, `strong_view_policy`가 선택한
  `aug_0` 또는 `aug_1`을 strong view로
  `weak_input_ids/strong_input_ids` batch key로 변환한다.
- legacy `weak_text/strong_text` row는 dataloader compatibility 경로에서만 허용한다.

## 제외

- unlabeled row 준비와 augmentation cache 생성
- LoRA model/tokenizer 생성
- Hydra config loading
- run artifact 저장

위 실행 조립은 `scripts/experiments/query_lora_ssl/query_ssl/`와 해당
entrypoint가 담당한다.
