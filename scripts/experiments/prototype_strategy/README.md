# Prototype Strategy Experiment

이 패키지는 `single`, `kmeans`, `dbscan` 기반 prototype 전략을 비교하고,
필요하면 static threshold policy까지 이어서 비교하는 실험 모듈이다.

## 읽기 순서

### 전략 비교 실험

1. `..prototype_strategy_experiment.py`
2. `runner.py`
3. `strategies.py`
4. `evaluation.py`
5. `scoring.py`
6. `projection.py`

### threshold sweep

1. `..prototype_threshold_sweep.py`
2. `sweep.py`
3. `threshold_policies.py`
4. `evaluation.py`
5. `scoring.py`

## 파일 역할

- `models.py`
  - 실험 결과와 중간 산출물 dataclass
- `strategies.py`
  - `shared` prototype build strategy를 실험용 `PrototypeIndex`로 감싸는 adapter
- `scoring.py`
  - prototype scorer runtime 설정과 scorer adapter
- `evaluation.py`
  - row별 score와 aggregate metric 계산
- `projection.py`
  - PCA/UMAP 시각화 산출물
- `threshold_policies.py`
  - static threshold policy 후보 구현

## 바로 조절 가능한 실험 축

- `runner.confidence_threshold`
- `runner.margin_threshold`
- `runner.scorer_backend_name`
- `runner.score_policy_name`
- `runner.score_top_k`
- `strategy.name`
- `strategy.kmeans_candidate_ks`
- `strategy.dbscan_eps_values`
- `strategy.dbscan_min_samples_values`
- `threshold_policies[*].thresholds`
- `threshold_policies[*].target_errors`

예시:

```bash
python -m scripts.experiments.prototype_strategy_experiment \
  runner.confidence_threshold=0.7 \
  runner.margin_threshold=0.05 \
  runner.score_policy_name=topk_mean_cosine \
  runner.score_top_k=2
```

```bash
python -m scripts.experiments.prototype_threshold_sweep \
  threshold_policies[0].thresholds=[0.7,0.8,0.9] \
  threshold_policies[1].target_errors=[0.03,0.05,0.1]
```

## newcomer 메모

- scorer 축을 바꾸고 싶으면 `scoring.py`의 `PrototypeScoringConfig`부터 본다.
- 새 builder 전략을 추가하고 싶으면 먼저
  `shared/src/services/prototypes/build_strategies.py`에 공용 계산을 추가하고,
  `strategies.py`에는 실험 adapter만 둔다.
- threshold와 scorer 관련 knob가 실제로 어디까지 열려 있는지는
  [docs/strategy_surface_map.md](......docs/strategy_surface_map.md)를
  같이 보면 빠르다.
- 이 패키지는 prototype 전략 비교 실험용이다.
  운영 runtime의 canonical prototype 계약은 `shared/src/contracts/prototype_contracts.py`가 기준이다.
