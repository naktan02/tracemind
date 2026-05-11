# USB FreeMatch

이 폴더는 Microsoft Semi-supervised-learning의 `semilearn/algorithms/freematch`
train-step core를 TraceMind Query SSL seam에 맞춰 둔다.

- upstream source: `microsoft/Semi-supervised-learning`
- upstream commit checked: `1ef4cbebcc0b368158315aeb425053858cf6c845`
- `train_step` 흐름은 FixMatch처럼 labeled CE, weak pseudo-label,
  strong consistency CE를 결합한다.
- FreeMatch 차이는 USB의 `FreeMatchThresholingHook`처럼 `time_p`,
  `p_model`, `label_hist` EMA state로 sample-wise adaptive threshold를 만들고,
  선택된 strong view에 entropy regularization을 더하는 점이다.
- TraceMind에서는 USB `AlgorithmBase`, distributed all-gather, checkpoint glue를
  제거하고 tensor-level threshold/entropy core만 method-local state로 유지한다.

입력 경계:

- 저장 row surface는 strict USB형 `text + aug_0 + aug_1`이다.
- dataloader가 `text`를 weak view로, 첫 번째 후보 `aug_0`을 strong view로
  `weak_input_ids/strong_input_ids` batch key로 변환한다.
- `aug_1`은 이후 strong view 2개를 쓰는 알고리즘을 위해 저장하지만 FreeMatch
  single strong-view 경로에서는 소비하지 않는다.
