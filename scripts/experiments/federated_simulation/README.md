# Federated Simulation

이 패키지는 `agent`와 `main_server` 코어를 직접 조합해 synthetic FL loop를
재현하는 실험층이다.

production runtime이 아니라, production service를 조립해서 검증하는
experiment package로 이해하면 된다.

중요:

- 이 패키지는 현재 로드맵에서 `시스템 FL 트랙`에 해당한다.
- `central LoRA classifier` 논문 비교선을 먼저 닫은 뒤, winner를 FL로 옮길 때
  이 패키지가 직접적인 기준 entrypoint가 된다.

현재 v1에서 이 패키지의 기본 비교선은
`embedding -> global classifier -> local interpretation`이다.
shared adapter와 prototype scoring은 비교 실험 축으로 함께 유지한다.

## 읽기 순서

1. `../run_federated_simulation.py`
   - Hydra entrypoint와 top-level config 조립
2. `simulation.py`
   - bootstrap -> client loop -> finalize 전체 orchestration
3. `task_config.py`
   - experiment config를 canonical `RoundOpenRequest`로 바꾸는 경계
4. `runtime.py`
   - prototype rebuild runtime, adapter seam, state load helper
5. `evaluation.py`
   - validation scoring과 training example 재구성
6. `sharding.py`
   - synthetic client split 규칙
7. `artifacts.py`
   - selection dump, manifest, prototype pack 저장

## 파일 역할

- `models.py`
  - simulation 전용 summary/config dataclass
- `simulation.py`
  - 가장 중요한 실행 흐름
- `runtime.py`
  - `main_server`/`agent` 코어 객체를 experiment용 저장소와 경로로 조립
- `task_config.py`
  - `main_server/src/services/rounds/models.py`의 canonical shape 재사용

## 바로 조절 가능한 실험 축

- `training_algorithm_profile`
- `round_runtime.adapter_family_name`
- `round_runtime.aggregation_backend_name`
- `round_runtime.classifier_head_bootstrap_logit_scale`
- `confidence_threshold`
- `margin_threshold`
- `training_task.objective.training_backend_name`
- `training_task.objective.algorithm_profile_name`
- `training_task.objective.example_generation_backend_name`
- `training_task.objective.evidence_backend_name`
- `training_task.objective.scorer_backend_name`
- `training_task.objective.score_policy_name`
- `training_task.objective.score_top_k`
- `training_task.objective.acceptance_policy_name`
- `training_task.objective.privacy_guard_name`
- `training_task.selection_policy.max_examples`
- `validation.scorer_backend_name`
- `validation.score_policy_name`
- `validation.score_top_k`

예시:

```bash
python -m scripts.experiments.run_federated_simulation \
  training_algorithm_profile=prototype_top1_confidence_v1 \
  confidence_threshold=0.7 \
  margin_threshold=0.0 \
  training_task.objective.privacy_guard_name=noop
```

주의:

- `aggregation_backend_name`과 `adapter_family_name`은 `round_runtime.*`로 노출된다.
- `weak_strong_pair` example backend는 source row에 weak/strong view가 이미 있어야 한다.
  현재 기본 JSONL row shape는 그 view를 따로 저장하지 않으므로, 별도 multiview row 공급이 없으면
  `prototype_rescore`를 계속 써야 한다.
- `weak_strong_pair`는 generic multiview input backend다.
  real agent의 stored scored event 경로는 아직 weak/strong view를 저장하지 않으므로
  현재는 simulation/row-source 경로가 우선이다.
- validation의 accepted_ratio는 raw score threshold가 아니라
  runtime과 같은 `evidence backend -> acceptance policy` 경로로 계산한다.
- `training_task.secure_aggregation.*` 필드는 실을 수 있지만,
  실제 secure aggregation/encryption runtime 실험이 되는 것은 아니다.

## newcomer 메모

- task shape를 바꾸고 싶으면 이 패키지부터 고치기보다
  `main_server/src/services/rounds/models.py`와 `task_config.py`를 같이 본다.
- validation scorer를 바꾸고 싶으면 `evaluation.py`와
  `shared/src/contracts/training_contracts.py`의 objective 축을 함께 본다.
- threshold, scorer, privacy knob의 현재 노출 범위는
  [docs/strategy_surface_map.md](../../../docs/strategy_surface_map.md)를
  먼저 보는 편이 빠르다.
