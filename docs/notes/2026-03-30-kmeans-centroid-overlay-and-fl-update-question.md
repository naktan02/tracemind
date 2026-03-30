# 2026-03-30 kmeans centroid overlay and FL update question

## User

1. 지금 k-means 비교 테스트 했었는데 기본 데이터셋에 대해서 분포를 사진으로 보여줬었는데 k-means 2개 선택해서 할 때 중심이 어디인지 그림에 찍히게 하고 싶어
2. fl 진행하는 코드 있잖아 근데 궁금한게 서버에서 무엇을 업데이트 하는거야? 임베딩 모델을 업데이트 하는건 아니잖아 프로토타입만 업데이트 진행하는건가? 그러면 진짜 fl처럼 모델이 업데이트 되는 방식은 아닌건가?

## Assistant

`tracemind-project` 스킬을 적용해서 진행합니다. 먼저 계획 문서와 현재 k-means 시각화 코드, FL 서버 업데이트 경로를 확인한 뒤 바로 필요한 수정과 설명을 정리하겠습니다.

## Assistant

계획 문서 순서를 먼저 확인했고, 이제 현재 단계와 사용자 요청이 맞물리는 지점을 확인 중입니다. 이어서 k-means 시각화 파일과 FL 오케스트레이션 경로를 함께 추적하겠습니다.

## Assistant

현재 활성 계획은 `personalized local inference + federated model improvement` 쪽으로 정리되어 있습니다. 이 기준에 맞춰, 시각화는 바로 수정하고 FL 질문은 실제 서버 코드 기준으로 정확히 답할 수 있게 관련 모듈을 읽고 있습니다.

## Assistant

계획은 이렇게 잡고 진행합니다.

1. `prototype_strategy` 시각화 경로에 centroid overlay를 넣습니다.
2. k-means 결과 중 실제 선택된 중심을 projection 좌표계로 투영해서 PNG와 JSONL에 같이 남기겠습니다.
3. 관련 단위 테스트를 추가하고, 필요한 범위만 실행해 회귀를 확인합니다.
4. 그 다음 FL 코드는 서버가 실제로 무엇을 업데이트하는지 코드 기준으로 정리해서 답변에 연결하겠습니다.

## Assistant

시각화 수정은 projection 단계에서 같은 reducer 좌표계로 prototype centroid를 다시 투영하는 방식으로 넣겠습니다. 이렇게 하면 실제 scatter 위에 현재 scoring에 쓰는 중심점을 바로 `X` 마커로 찍을 수 있고, k-means가 `k=2`일 때도 두 중심이 그대로 보입니다.

## Assistant

코드 반영은 끝냈고, 지금은 줄 길이와 데이터 모델 연결이 어색한 부분이 없는지 확인한 뒤 테스트를 돌리고 있습니다. FL 설명은 이미 핵심 경로를 확인했기 때문에, 검증만 끝나면 서버가 실제로 갱신하는 객체를 코드 기준으로 정리해서 답변하겠습니다.

## Assistant

`python` 심볼릭이 없는 환경이라 `python3`로 다시 포맷과 테스트를 돌리겠습니다. 구현 자체는 반영되어 있고, 지금은 실행 환경 차이만 정리하는 단계입니다.

## Assistant

`uv` 자체는 있지만 기본 캐시 경로가 read-only라서, 캐시만 `/tmp`로 돌려서 검증을 계속합니다. 이건 코드 문제가 아니라 샌드박스 쓰기 제한입니다.

## Assistant

`pytest` 수집 오류는 패키지 경로가 안 잡혀서 생긴 것이고, `PYTHONPATH`와 extras를 명시하면 해결될 가능성이 큽니다. 그 설정으로 린트와 테스트를 다시 확인하겠습니다.

## Assistant

구현과 검증은 끝났습니다. 마지막으로 변경 사항을 정리하고, 프로젝트 규칙에 맞춰 이번 대화 기록을 `docs/notes/`에 남긴 뒤 바로 답변을 마무리하겠습니다.

## Assistant

최종 답변 요지:

1. `scripts/experiments/prototype_strategy/projection.py`에서 PCA/UMAP projection 위에 k-means prototype centroid를 같은 좌표계로 투영해 `X` 마커와 짧은 라벨로 그리도록 수정했다.
2. `scripts/experiments/prototype_strategy/runner.py`는 전략 평가 후 `kmeans` prototype index를 projection 단계로 넘기도록 바뀌었다.
3. `scripts/experiments/prototype_strategy/models.py`의 `ProjectionArtifact`와 `summary.json`에는 어떤 prototype 전략 overlay를 썼는지와 centroid 좌표 JSONL 경로가 같이 남는다.
4. `tests/unit/test_prototype_projection_service.py`를 추가해서 projected centroid 파일 생성과 figure 산출을 검증했다.
5. 검증은 `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev --extra experiments ruff check ...` 와 `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache uv run --extra dev --extra experiments pytest ...` 기준으로 통과했다.
6. FL 코드는 현재 서버가 `PrototypePack`만 업데이트하는 구조가 아니다. 현재 구현은 `VectorAdapterState`라는 경량 전역 adapter state를 집계해 새 `model_revision`을 발행하고, 그 새 adapter state에 맞춰 `PrototypePack`을 다시 빌드하는 구조다.
7. 따라서 지금 구현은 full embedding backbone 자체를 직접 미세조정하는 FL은 아니지만, 전역 모델 파라미터의 일부인 adapter state를 연합학습으로 업데이트하는 형태라고 정리할 수 있다.

## User

1. 해당 fl 만든 구조 보려면 파일 어떤것들 봐야해?
2. adapter 파라미터로 무엇을 하는건데?
3. k-means로 한 2개 중심 찍힌거 저장된거야? 아니면 내가 실행해야해?
4. 지금 카테고리 마다 2개씩 중심이 있어도 현재 구조에서 fl 하는데 문제 없어? 들어오는 값에 맞게 잘 조정 가능해?
5. 현재 fl로 했을 떄 무엇이 변하는거야? 카테고리 중심 위치? 각 쿼리가 들어올 때 임베딩 되는 위치? 어떤 값이 들어왔을 때 확률이 높으면 이 문장은 어떤 카테고리다라고 라벨링하고 학습하잖아 그걸 서버로 보내는거고 학습을 하면 무엇을 학습하는거고 무엇이 변화하는거지?

## Assistant

핵심부터 말하면, 현재 FL 경로는 `kmeans multi-prototype runtime`이 아니라 `single centroid runtime + adapter update`라고 답했다. 그래서 실험용 k-means 2중심과 현재 FL runtime은 아직 같은 구조가 아니라고 설명했다.

답변 요지:

1. FL 구조를 보려면 `scripts/experiments/run_federated_simulation.py`, `agent/src/services/training/local_training_service.py`, `agent/src/services/training/pseudo_label_service.py`, `main-server/src/services/rounds/round_manager_service.py`, `main-server/src/services/rounds/aggregation_service.py`, `shared/src/domain/entities/training/vector_adapter_state.py`, `shared/src/domain/entities/training/vector_adapter_delta.py`, `shared/src/contracts/prototype_contracts.py`, `scripts/prototypes/prototype_pack_builder.py` 순으로 보면 된다고 안내했다.
2. 현재 adapter 파라미터는 `VectorAdapterState.dimension_scales`라는 차원별 scale 벡터이고, base embedding이 나온 뒤 각 차원에 scale을 곱하고 다시 normalize하는 방식이라고 설명했다.
3. k-means 2개 중심이 찍힌 실제 그림은 아직 새로 생성한 것이 아니라 코드만 바뀐 상태이고, 실제 PNG와 JSONL은 다시 실행해야 저장된다고 설명했다.
4. 카테고리마다 2개 중심은 실험 코드에서는 되지만 현재 active FL/runtime contract는 카테고리당 centroid 1개 구조라서 그대로는 맞지 않는다고 설명했다.
5. 현재 FL에서 변하는 것은 base encoder 자체가 아니라 adapter state와, 그 adapter state를 반영해 다시 계산한 prototype centroid라고 설명했다.

## User

1. k-means 찍히는것까지 실행하려고 하면 실행 코드어떻게 쳐야해?
2. 그러면 현재 fl은 쿼리가 들어오면 임베딩하고 그 후에 어뎁터를 만나 보정을 진행하고 프로토타입과 비교인건가?

## Assistant

실행 커맨드는 실험 설정과 Hydra override 문법까지 같이 확인해서 바로 복붙 가능한 형태로 정리하겠다고 답했다. 이어서 두 번째 질문에는 현재 scoring 흐름을 정확한 경로로 설명하겠다고 했다.

후속 답변 요지:

1. `prototype_strategy_experiment`를 실행하면 projection 그림과 centroid 좌표 JSONL이 같이 생긴다고 설명했다.
2. 당시 구조에서는 `k=2` 강제를 위해 `runner.strategies` 전체를 override하는 긴 Hydra 명령이 필요하다고 설명했다.
3. 현재 FL 흐름은 `텍스트 -> base embedding -> adapter 보정 -> prototype과 cosine 비교 -> pseudo-label 선택 -> adapter delta 생성 -> 서버 집계 -> 새 adapter state 발행 -> prototype 재생성`이라고 설명했다.
4. 다만 active FL runtime은 `single centroid` 구조라고 설명했다.

## User

왜 이렇게 실행해? 나는 보통 스크립트에서 하이드라를 통한 실행 쉽게 만들어 놓지 않았나?

uv run --extra experiments python scripts/experiments/prototype_strategy_experiment.py \
  'runner.strategies=[{_target_:scripts.experiments.prototype_strategy.strategies.KMeansPrototypeStrategy,candidate_ks:[2],silhouette_sample_size:2000,random_state:42}]'

## Assistant

지적이 맞다고 답했다. 현재 `prototype_strategy_experiment` config가 `runner.strategies` 객체 리스트 안에 전략 설정을 묶어 둔 구조라 override ergonomics가 나쁘다고 설명했다. 반면 `prototype_threshold_sweep`는 `strategy.name`, `strategy.kmeans_candidate_ks`처럼 top-level 전략 설정으로 분리돼 있어 Hydra override가 자연스럽다고 정리했다.

## User

지금 구조가 잘못된 방식인거지? 리펙터링 진행하는게 더 좋은거지?

## Assistant

완전히 틀린 건 아니지만 현재 저장소 기준에서는 비일관적이고 불편한 구조라서 리팩터링이 맞다고 답했다. `prototype_strategy_experiment`도 `threshold_sweep`처럼 `strategy` 블록으로 빼는 방향이 더 낫다고 설명했다.

## User

진행해

## Assistant

이번 리팩터링에서는 다음을 수행했다.

1. `scripts/conf/experiments/prototype_strategy.yaml`
   - `runner.strategies` 객체 리스트를 제거했다.
   - top-level `strategy` 블록을 추가했다.
   - 기본값은 `strategy.name=all`로 두었다.

2. `scripts/experiments/prototype_strategy/runner.py`
   - `PrototypeExperimentRunner`가 config 내부 전략 리스트 대신 request의 `strategies`를 받도록 바꿨다.
   - `build_strategy`, `build_strategies` helper를 추가했다.
   - `all` 모드에서는 `single`, `kmeans`, `dbscan` 세 전략을 생성하고, 단일 전략 모드에서는 하나만 생성하게 했다.
   - projection overlay는 `kmeans`가 있으면 `kmeans`, 없으면 selected strategy를 사용하도록 보강했다.

3. `scripts/experiments/prototype_strategy/main.py`
   - top-level `strategy` 설정으로부터 `build_strategies(...)`를 호출해 request에 넣도록 바꿨다.

4. `scripts/experiments/prototype_strategy/__init__.py`
   - `build_strategy`, `build_strategies`를 export에 추가했다.

5. `scripts/experiments/prototype_strategy_experiment.py`
   - thin entrypoint export에도 `build_strategy`, `build_strategies`를 추가했다.

6. `tests/unit/test_scripts_hydra_configs.py`
   - `strategy.name=kmeans`, `strategy.kmeans_candidate_ks=[2]` short override가 compose되는지 검증하도록 갱신했다.

7. `tests/unit/test_prototype_strategy_experiment.py`
   - `build_strategies`가 `kmeans` 단일 전략과 `all` 전략 세트를 올바르게 구성하는지 확인하는 테스트를 추가했다.

8. `scripts/README.md`
   - 새 Hydra 실행 예시로 `strategy.name=kmeans strategy.kmeans_candidate_ks=[2]`를 추가했다.

검증:

1. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra dev --extra experiments ruff check ...`
2. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra dev --extra experiments pytest tests/unit/test_scripts_hydra_configs.py tests/unit/test_prototype_strategy_experiment.py tests/unit/test_prototype_projection_service.py tests/unit/test_run_federated_simulation.py`
3. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra experiments python scripts/experiments/prototype_strategy_experiment.py strategy.name=kmeans strategy.kmeans_candidate_ks=[2] runtime=cpu_local embedding=hash_debug embedding.hash_dim=64 --cfg job`

위 검증은 모두 통과했다.

## User

(tracemind-server) jmgjmg102@DESKTOP-VVOL8MH:~/tracemind_server$   uv run python scripts/experiments/prototype_strategy_experiment.py \
    strategy.name=kmeans \
    strategy.kmeans_candidate_ks=[2]
Error executing job with overrides: ['strategy.name=kmeans', 'strategy.kmeans_candidate_ks=[2]']
Traceback (most recent call last):
  File "/home/jmgjmg102/tracemind_server/scripts/experiments/prototype_strategy/main.py", line 29, in main
    output_dir = config.output.base_dir / run_id
                 ~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~
TypeError: unsupported operand type(s) for /: 'str' and 'str'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

## Assistant

실행 오류는 Hydra가 넘겨주는 경로 값이 문자열인데 entrypoint가 `Path`처럼 `/` 연산을 해서 생긴 타입 문제라고 설명했다. 이 패턴이 `threshold_sweep`에도 같은 형태로 있어서 함께 손보겠다고 답했다.

## Assistant

수정 내용:

1. `scripts/experiments/prototype_strategy/io_utils.py`
   - `resolve_output_dir(base_dir, run_id)`를 추가했다.
   - `load_jsonl_rows`, `dump_json`, `dump_jsonl`이 `str | Path`를 모두 받도록 바꿨다.

2. `scripts/experiments/prototype_strategy/main.py`
   - `output_dir = resolve_output_dir(config.output.base_dir, run_id)`로 변경했다.

3. `scripts/experiments/prototype_strategy/sweep_main.py`
   - 같은 방식으로 `resolve_output_dir(...)`를 사용하게 바꿨다.

4. `scripts/conf/dataset/ourafla.yaml`
   - `test_jsonl: ${dataset.test_labeled_jsonl}`를 추가했다.

5. `scripts/conf/dataset/cssrs.yaml`
   - `test_jsonl: null`을 추가했다.

6. `tests/unit/test_prototype_strategy_experiment.py`
   - `resolve_output_dir`가 문자열 base_dir를 처리하는지,
   - `load_jsonl_rows`가 문자열 path를 처리하는지 검증하는 테스트를 추가했다.

7. `tests/unit/test_scripts_hydra_configs.py`
   - `dataset.test_jsonl == dataset.test_labeled_jsonl` 검증을 추가했다.

검증:

1. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra dev --extra experiments pytest tests/unit/test_scripts_hydra_configs.py tests/unit/test_prototype_strategy_experiment.py`
2. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra experiments python scripts/experiments/prototype_strategy_experiment.py strategy.name=kmeans strategy.kmeans_candidate_ks=[2] runtime=cpu_local embedding=hash_debug embedding.hash_dim=64`
3. 실제 실행 결과:
   - `output_dir=data/processed/evaluations/prototype_strategy_experiments/20260330T141238Z`
   - `selected_strategy=kmeans`
   - projection 산출물: `train_pca.png`, `train_umap.png`, `train_pca.kmeans_prototypes.jsonl`, `train_umap.kmeans_prototypes.jsonl`

## User

데이터셋은 train으로 ㄴ하는건가?

## Assistant

아니라고 답했다. 현재 구조는 `train`만 쓰는 게 아니라, projection과 prototype 생성은 `train`, 전략 선택은 `validation`, 최종 평가 요약은 `test`를 사용한다고 설명했다.

## User

해당 중심점은 최적이라고 생각하는 곳에 k-means 돌려서 나온 결과인건가? 그리고 pca와 umap의 차이는?
또 지금 k-mean안한고 싱글로도 실행해보련면 어떻게 하지?
readme는 업데이트 안해?

## Assistant

최종 답변 요지:

1. 지금 그림에 찍히는 중심점은 사람이 수동으로 찍은 점이 아니라, train split 임베딩에 대해 k-means를 돌린 뒤 선택된 `k`에 대해 다시 전체 train 임베딩으로 fit해서 나온 cluster centroid를 prototype으로 만든 결과다.
2. 다만 그림 위 좌표는 원공간 centroid 자체가 아니라, 그 centroid를 PCA/UMAP 2차원 좌표계에 다시 투영한 시각화용 위치다.
3. `PCA`는 선형 투영이라 전체적인 큰 분산 구조를 보기에 좋고, `UMAP`은 비선형 투영이라 국소 cluster 구조를 더 잘 보여주지만 축 자체의 해석 가능성은 낮다고 설명했다.
4. `single`만 실행하려면 `uv run python scripts/experiments/prototype_strategy_experiment.py strategy.name=single`로 실행하면 된다고 안내했다.
5. README는 이번에 실제로 업데이트했다. `single` 실행 예시, `kmeans k=2` 실행 예시, 그리고 projection=train / selection=validation / final metric=test 설명을 추가했다.

## User

20260330T142022Z에서 umap와 pca봐봐 왜 중심값이 저런곳에 찍힌거지 저러니까 2개일떄보다 안 좋은거 아닌가 20260330T141418Z에서 k-means로 한것도 중심이 왜 저기에 위치하는지 이해가 안가네

중심 구하는 공식이 잘못된거 아냐?

## Assistant

현재 그림에 찍히는 점은 `sklearn.cluster_centers_`나 2D 시각적 중심이 아니라, runtime scoring에 쓰는 normalized prototype을 PCA/UMAP에 다시 투영한 값이라고 설명했다. 그래서 시각화 관점에서는 오해를 부르는 표현이고, 특히 UMAP처럼 비선형 투영에서는 실제 2D 점 무리의 가운데와 크게 다를 수 있다고 답했다.

또 `20260330T141418Z`의 `k=2` run은 silhouette가 매우 낮아서 애초에 분리가 약하다고 설명했고, `20260330T142022Z` single run에서는 UMAP 기준 `normal`의 runtime prototype 위치와 실제 2D 점 평균 위치가 꽤 다르다는 점을 확인했다고 말했다.

정리하면 scoring 공식 자체가 틀렸다기보다, 현재 그림이 다른 의미의 중심을 중심처럼 보이게 만들어 혼동을 줬다고 답했다. 그래서 다음 단계로 `runtime prototype`과 `2D visual center`를 분리해 보여주는 방향을 제안했다.

## User

runtime prototype이 뭔데?

## Assistant

`runtime prototype`은 실제 추론에서 쿼리 임베딩과 cosine similarity를 계산하는 대표 벡터라고 설명했다. `single`이면 label 내 임베딩 평균 후 정규화한 벡터이고, `kmeans`면 cluster별 임베딩 평균 후 정규화한 벡터라고 답했다.

즉 그림 예쁘게 보이기 위한 점이 아니라, 실제 scoring 기준점이라고 설명했다.

## User

근데 단순히 single일때도 평균하면 안 좋은거 아냐? k-means로 한다고 해도 1개로 하면 더 좋은 중심을 하는거 아냐?

## Assistant

`single`은 baseline일 뿐이고, 클래스 내부가 여러 모드면 안 좋아질 수 있다고 설명했다. 다만 현재 구조에서 `kmeans(k=1)`은 사실상 전체 평균 하나를 쓰는 점에서 `single`과 거의 같다고 답했다. 개선은 `k>=2`처럼 multi-prototype일 때 생긴다고 정리했다.

## User

지금거 추가로  runtime prototype말고 시각적 중앙 점도 새로운 파일로 해서 볼 수 있게 되나? 그러면 진짜 2d에서 어느지점에 위치한지 볼 수 있나? 그리고 그렇게 하려면 어떤 공식을 써야 하는거야?

## Assistant

가능하다고 답했고, `runtime prototype`은 그대로 두되 `2D visual center`를 별도 JSONL과 그림 오버레이로 추가하겠다고 설명했다.

계산식은 다음처럼 정리했다.

1. `single` 또는 label 단위 center:
   - 투영된 2D 점들에 대해 label별 산술평균
   - `visual_center(label) = (1/n) * sum_i projection(x_i)`

2. `kmeans` visual center:
   - 원공간 임베딩에서 label별 `kmeans` cluster 할당을 다시 계산
   - 각 cluster에 속한 샘플들의 2D 투영점 평균
   - `visual_center(label, cluster) = (1/m) * sum_{i in cluster} projection(x_i)`

이 값은 실제 그림 위에 찍힌 점들의 중심이므로, 사용자가 기대한 "2D에서 어느 위치인지"를 보는 용도에 더 맞는다고 설명했다.

## Assistant

수정 내용:

1. `scripts/experiments/prototype_strategy/models.py`
   - `ProjectionArtifact.visual_center_points_path` 필드를 추가했다.
   - `summary.json`에 `visual_center_points_path`를 기록하도록 `ExperimentSummary.to_dict()`를 확장했다.

2. `scripts/experiments/prototype_strategy/projection.py`
   - `visual_center_points` 계산 경로를 추가했다.
   - `kmeans`일 때는 label별 `selected_k`와 동일한 cluster 할당을 원공간에서 다시 계산한 뒤, 샘플된 2D 좌표 평균으로 `visual center`를 만든다.
   - `single`/기타 전략일 때는 label별 2D 좌표 평균을 `visual center`로 만든다.
   - 결과를 `train_{pca|umap}.{strategy|label}_visual_centers.jsonl`에 저장한다.
   - 그림에도 diamond marker(`D`)로 `visual center`를 겹쳐 그리도록 바꿨다.

3. `tests/unit/test_prototype_projection_service.py`
   - `kmeans` projection에서 prototype JSONL과 visual center JSONL이 모두 생성되는지 검증하는 테스트를 추가했다.
   - `single` 전략에서 visual center 좌표가 실제 2D 점 평균과 일치하는지 검증하는 테스트를 추가했다.

4. `scripts/README.md`
   - projection 산출물 목록에 `*_visual_centers.jsonl`을 추가했다.
   - `*_prototypes.jsonl`은 runtime scoring용 prototype의 2D 투영,
     `*_visual_centers.jsonl`은 그림 위 점들의 산술평균 중심이라는 설명을 추가했다.

검증:

1. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra dev --extra experiments pytest tests/unit/test_prototype_projection_service.py tests/unit/test_prototype_strategy_experiment.py tests/unit/test_scripts_hydra_configs.py`
   - 14개 테스트 통과

2. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra dev --extra experiments ruff check scripts/experiments/prototype_strategy/projection.py tests/unit/test_prototype_projection_service.py`
   - 통과

3. `PYTHONPATH=. UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mpl uv run --extra experiments python scripts/experiments/prototype_strategy_experiment.py strategy.name=kmeans strategy.kmeans_candidate_ks=[2] runtime=cpu_local embedding=hash_debug embedding.hash_dim=64`
   - 실행 성공
   - `output_dir=data/processed/evaluations/prototype_strategy_experiments/20260330T143736Z`
   - `selected_strategy=kmeans`
   - `kmeans: accuracy=0.2620, accepted_ratio=0.3034, prototype_count=8`
   - `test: accuracy=0.2510, accepted_ratio=0.3206`

4. 생성 산출물:
   - `data/processed/evaluations/prototype_strategy_experiments/20260330T143736Z/projections/train_pca.kmeans_visual_centers.jsonl`
   - `data/processed/evaluations/prototype_strategy_experiments/20260330T143736Z/projections/train_umap.kmeans_visual_centers.jsonl`
   - 기존 prototype 파일도 그대로 유지

예시로 `train_umap.kmeans_visual_centers.jsonl`에는 아래처럼 category/cluster별 2D 중심이 저장된다.

- anxiety cluster 0: `(2.0223, 4.7187)`
- anxiety cluster 1: `(3.0963, 5.6929)`
- normal cluster 0: `(3.5025, 5.9224)`
- normal cluster 1: `(2.0726, 5.2244)`

같은 run의 `train_umap.kmeans_prototypes.jsonl`에는 runtime prototype의 2D 투영이 별도로 저장된다. 둘을 비교하면 "실제 scoring 기준점"과 "그림 위 시각적 중심"이 어떻게 다른지 직접 확인할 수 있다.
