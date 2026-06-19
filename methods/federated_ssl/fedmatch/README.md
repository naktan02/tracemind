# FedMatch

`fedmatch/`는 FedMatch method의 identity, local objective, partition 의미,
helper policy, server update policy를 소유한다. Simulation, agent, main_server는
FedMatch 이름으로 직접 분기하지 않고 이 package의 descriptor와 capability plan을
읽는다.

## What It Owns

- FedMatch method descriptor와 report metadata
- `labels-at-client`, `labels-at-server` scenario default
- agreement pseudo-labeling, confidence filtering, helper consistency loss
- `sigma` / `psi` partition 이름, loss routing, original parameter snapshot
- helper selection policy와 peer context parameter 해석
- FedMatch-style partitioned server update policy metadata

## Read Path

FedMatch 동작을 볼 때는 아래 네 파일만 먼저 읽는다.

1. `descriptor.py`: method identity, required capability, runtime entrypoint
2. `local_objective.py`: agreement pseudo-label, confidence filter, sigma/psi loss
3. `partitioning.py`: sigma/psi partition mapping과 runtime plan
4. `method_surface.py`: scenario default, report metadata, helper/server policy

필요할 때만 `original_spec.py`, `compatibility.py`, `helper_selection.py`,
`client_diagnostics.py`를 따라간다. `local_runtime.py`는 descriptor가 호출하는 얇은
entrypoint이며, FedMatch config를 partition runtime plan으로 정규화한 뒤 PEFT text
encoder runtime bridge로 넘긴다.

## Key Concepts

FedMatch에서 보존해야 하는 핵심 의미는 특정 LoRA 구현이 아니라 parameter
decomposition이다.

| Concept | Meaning |
| --- | --- |
| `sigma` | labeled/supervised objective가 업데이트하는 partition |
| `psi` | unlabeled agreement, helper consistency, L1/L2 regularization이 업데이트하는 partition |
| `lambda_l1` | `psi` partition에만 적용되는 sparsity regularization |
| `lambda_l2` | 같은 trainable scope의 `sigma`와 `psi` 차이에 적용되는 regularization |
| helper context | 이전 round client snapshot과 probe output 기반 nearest-neighbor helper signal |
| partitioned update | merged delta가 아니라 partition별 trainable adapter/head delta를 server가 해석하는 update |

physical forward는 `psi` 단독 logits가 아니라 trainable parameter를 `sigma + psi`로
합성한 effective state를 기준으로 계산한다. optimizer 대상만 supervised step은
`sigma`, unsupervised step은 `psi`로 제한한다.

## Current Runtime Surface

현재 실행 surface는 PEFT text encoder update family 위에서 FedMatch partition 의미를
해석한다.

- family-specific bridge와 partitioned optimizer loop는
  `methods/adaptation/peft_text_encoder/federated_ssl/`가 소유한다.
- FedMatch method package는 partition 이름, objective, helper/server policy 의미를
  계속 소유한다.
- original FedMatch snapshot은 `wyjeong/FedMatch`
  `4947aa255d59bd37915e25a719763aaaf5d7e067` 기준으로 해석한다.
- `method_owned` 실행에서는 descriptor가 파생한 `fedmatch_partitioned` policy에 따라
  runtime adapter가 PEFT encoder `partitioned_delta_average` backend로
  `partitioned_deltas`를 소비한다.
- `fedmatch_agreement` local SSL policy는 generic Hydra leaf로 직접 고르지 않고
  `fssl_method=fedmatch`와 `ssl_method.scenario` 조합에서 파생한다.

Simulation에는 helper weak-view probability provider, `labels-at-server` client-local
`psi` upload slice, supervised seed server step, sparse S2C/C2S projection, previous
partition snapshot accounting이 연결되어 있다. 실제 네트워크 packet 측정은 아직
artifact estimate로 남긴다.

## Boundaries

- PEFT/LoRA/DoRA physical adapter materialization은
  `methods/adaptation/peft_text_encoder/`와 `methods/adaptation/peft_adapters/`가
  소유한다.
- FL round lifecycle, update acceptance, publication은 `main_server`가 소유한다.
- Simulation orchestration과 report artifact 저장은
  `scripts/experiments/fl_ssl/`가 소유한다.
- shared contract에는 `sigma`, `psi`, `fedmatch` 같은 method-local 의미를 넣지 않는다.
  필요한 경우 partition mapping과 artifact reference 같은 canonical shape만 둔다.
- `server_step_policy`는 server-side 추가 학습 여부이고, `server_update_policy`는
  client delta를 server가 어떤 의미로 해석할지다. 둘은 같은 축이 아니다.
