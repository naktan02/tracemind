# USB SoftMatch

이 폴더는 Microsoft Semi-supervised-learning의 `semilearn/algorithms/softmatch`
train-step core를 TraceMind Query SSL seam에 맞춰 둔다.

- upstream source: `microsoft/Semi-supervised-learning`
- upstream commit checked: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- `train_step` 흐름은 labeled CE, weak-view pseudo-label, strong-view consistency
  CE를 결합한다.
- SoftMatch 차이는 distribution alignment가 적용된 weak probability로 truncated
  Gaussian sample weight를 계산하되, pseudo-label target은 원본 weak logits에서
  만든다는 점이다.
- TraceMind에서는 USB `AlgorithmBase`, distributed all-gather, checkpoint glue를
  제거하고 tensor-level EMA distribution alignment와 weighting state만 method-local
  state로 유지한다.

입력 경계:

- 저장 row surface는 strict USB형 `text + aug_0 + aug_1`이다.
- dataloader가 `text`를 weak view로, `strong_view_policy`가 선택한 `aug_0` 또는
  `aug_1`을 strong view로 `weak_input_ids/strong_input_ids` batch key로 변환한다.
