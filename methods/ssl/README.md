# SSL Methods

`methods/ssl/`는 FixMatch, FlexMatch, FreeMatch, UDA, pseudo-label self-training 같은
semi-supervised learning objective framework surface를 둔다.

`methods/ssl/algorithms/`에는 method-specific step 계산과 algorithm descriptor를
둔다.
`methods/ssl/hooks/`는 pseudo-label target 생성, confidence mask, threshold 기반
selection rule처럼 중앙 SSL과 FL SSL client가 공유하는 subroutine을 소유한다.
`methods/ssl/teacher_pseudo_label.py`는 teacher prediction을 pseudo-label row,
selection metadata, diagnostics payload로 해석하는 method-neutral 의미를 소유한다.
`methods/ssl/base.py`와 `methods/ssl/registry.py`는 공통 framework surface다.
Dataset loading, Hydra 조립, artifact 저장은 `scripts/`가 맡는다.

새 Query SSL algorithm을 추가할 때는 먼저 `NEW_METHOD.md`를 본다. 같은
`compute_step(model, labeled_batch, unlabeled_batch)` seam으로 표현되는 방법론은
`methods/ssl/algorithms/<method>/`와
`conf/strategy_axes/ssl_objective/consistency_method/<method>_v1.yaml`을 추가하는 것이 기본이다.

Algorithm은 USB/SemiLearn처럼 role별 hook 교체가 가능해야 한다. TraceMind는
mutable hook dict와 문자열 lookup 대신 `SslObjectiveHooks` typed bundle로
pseudo-labeling, masking, consistency loss role을 명시한다. Algorithm-local hook은
처음에는 해당 `algorithms/<method>/`에 두고, 두 개 이상 algorithm에서 안정적으로
공유될 때만 `methods/ssl/hooks/`로 승격한다.

새 SSL 방법론을 추가할 때는 먼저 기존 축 조합으로 표현되는지 확인한다.

- view requirement
- supervised branch
- pseudo-label generator
- confidence selector 또는 weighting
- consistency loss
- distribution alignment
- teacher 또는 memory state
- mix strategy
- adversarial perturbation

기존 축 조합이면 `methods/ssl/algorithms/<method>/`와 Hydra leaf, 테스트만 추가한다.
새 primitive가 필요하면 method-local helper에 숨기지 말고 `methods/ssl/hooks/` 등
가장 가까운 공통 owner에 축 구현을 함께 추가한다. FedProx처럼 SSL method가 아니라
local training loss에 붙는 regularizer는 `methods/adaptation/local_objective_regularizers/`
가 소유한다. SCAFFOLD/FedDyn처럼 FL round/server/client state를 요구하는 방법론은
`methods/ssl` 내부 hook으로 처리하지 않고 `methods/federated_ssl/` capability 분리를
함께 진행한다.
