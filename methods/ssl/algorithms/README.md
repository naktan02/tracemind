# SSL Algorithms

`methods/ssl/algorithms/`는 PseudoLabel, PiModel, MeanTeacher, FixMatch, ReFixMatch,
FlexMatch, FreeMatch, AdaMatch, Dash, UDA, SoftMatch, CoMatch, SimMatch, MixMatch,
ReMixMatch 같은 SSL objective 구현을 method-specific module로 둔다.

공유 가능한 tensor/module primitive는 `methods/ssl/primitives/`에 둔다.
교체 가능한 pseudo-label, masking, thresholding, distribution alignment role은
`methods/ssl/hooks/`에 둔다.
USB weak/strong train-step glue처럼 algorithm package 내부에서만 공유되는
흐름은 `usb_consistency.py`에 둔다.
단, method-local package가 다른 method-local package의 helper를 직접 빌려 쓰지
않는다. MixUp, sharpening, soft-target loss처럼 여러 method에서 같은 의미가 된
helper는 `primitives/`로 올리고, method별 수식 차이가 남아 있으면 각 method-local에
둔다.
Hydra 조립, dataset loading, augmentation materialization, artifact 저장은
`scripts/` runtime adapter가 담당한다.

새 algorithm 추가 절차와 비-USB 계열 판단 기준은 `../NEW_METHOD.md`를 따른다.
