# SSL Algorithms

`methods/ssl/algorithms/`는 PseudoLabel, PiModel, MeanTeacher, FixMatch, ReFixMatch,
FlexMatch, FreeMatch, AdaMatch, Dash, UDA, SoftMatch, CoMatch, SimMatch, MixMatch 같은
SSL objective 구현을 method-specific module로 둔다.

공유 가능한 tensor-level subroutine은 `methods/ssl/hooks/`에 둔다.
USB weak/strong train-step glue처럼 algorithm package 내부에서만 공유되는
흐름은 `usb_consistency.py`에 둔다.
단, `hook.compute_loss(...)`, `hook.generate_targets(...)`, `compute_prob(x.detach())`
같은 한 줄 wrapper는 만들지 않는다. 공통 helper는 required view, loader validation,
tokenized weak/strong forward처럼 batch contract나 실행 순서를 실제로 숨기는 경우에만
둔다.
Hydra 조립, dataset loading, augmentation materialization, artifact 저장은
`scripts/` runtime adapter가 담당한다.

새 algorithm 추가 절차와 비-USB 계열 판단 기준은 `../NEW_METHOD.md`를 따른다.
