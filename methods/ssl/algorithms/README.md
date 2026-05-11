# SSL Algorithms

`methods/ssl/algorithms/`는 PseudoLabel, FixMatch, FreeMatch, R-Drop, MixText 같은
SSL objective 구현을 method-specific module로 둔다.

공유 가능한 tensor-level subroutine은 `methods/ssl/hooks/`에 둔다.
USB weak/strong train-step glue처럼 algorithm package 내부에서만 공유되는
흐름은 `usb_consistency.py`에 둔다.
Hydra 조립, dataset loading, augmentation materialization, artifact 저장은
`scripts/` runtime adapter가 담당한다.
