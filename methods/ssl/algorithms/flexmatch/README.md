# USB FlexMatch

이 폴더는 Microsoft Semi-supervised-learning의 `semilearn/algorithms/flexmatch`
train-step core를 TraceMind Query SSL seam에 맞춰 둔다.

- upstream source: `microsoft/Semi-supervised-learning`
- upstream commit checked: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- `train_step` 흐름은 FixMatch와 동일하게 labeled CE, weak pseudo-label,
  strong consistency CE를 결합한다.
- FlexMatch 차이는 USB의 `FlexMatchThresholdingHook`처럼 `idx_ulb`별
  `selected_label` state와 classwise adaptive threshold를 유지하는 점이다.
- TraceMind에서는 unlabeled dataloader의 stable `row_indices`가 USB `idx_ulb`에
  대응한다.

입력 경계:

- 저장 row surface는 strict USB형 `text + aug_0 + aug_1`이다.
- dataloader가 `text`를 weak view로, `strong_view_policy`가 선택한
  `aug_0` 또는 `aug_1`을 strong view로
  `weak_input_ids/strong_input_ids` batch key로 변환한다.
