# SSL Algorithms

`methods/ssl/algorithms/`는 PseudoLabel, FixMatch, R-Drop, MixText 같은 SSL objective
구현을 method-specific module로 둔다.

공유 가능한 tensor-level subroutine은 `methods/ssl/hooks/`에 둔다.
Hydra 조립, dataset loading, augmentation materialization, artifact 저장은
`scripts/` runtime adapter가 담당한다.
