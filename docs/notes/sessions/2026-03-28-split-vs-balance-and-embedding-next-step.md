## User

• prototype seeding까지 smoke test 기준으로 닫혔습니다. 이제 final에서는 왜 균형화보다 split 유지가 맞는지, 그리고
  지금 생성된 pack이 debug backend라는 점을 구분해서 설명하겠습니다.

이거 너가 설명하다가 내가 실수로 컨트롤z눌러서 없애버렸는데 맥락알고 있어?

그리고 임베딩 모델 다운로드 해서 split 나눈 train 데이터로 카테고리 임베딩 먼저 하고 시작하면 다음단계 더 수월한거 아냐?

## Assistant

현재 저장소 기준으로는 잃어버린 문장을 그대로 복원할 수는 없지만, 설명이 놓여 있던 맥락은 재구성할 수 있다.

1. prototype seeding은 smoke test 수준에서는 닫혔다.
2. final에서는 두 가지를 분리해서 설명하는 것이 맞다.
   - 왜 데이터 균형화보다 split 유지가 우선인지
   - 지금 생성된 pack은 semantic 품질 검증용 실모델 pack이 아니라 debug backend 기반 pack인지
3. 현재 생성물 기준 사실관계는 아래와 같다.
   - split manifest는 `data/processed/splits/ourafla_mental_health_text_classification.v1.manifest.json`
   - train/validation은 stratified split으로 나뉘어 있다.
   - prototype pack manifest는 `data/processed/prototype_packs/proto_ourafla_hashdebug_v1.manifest.json`
   - 이 manifest의 `input_jsonl`은 train split이며, `backend`는 `hash_debug`, `embedding_model_id`는 `hash_debug/mxbai_shape_proxy`다.

final에서 설명해야 하는 핵심 논리는 아래다.

### 왜 균형화보다 split 유지가 맞는가

1. split의 목적은 학습용 데이터와 검증용 데이터를 분리해서 leakage 없이 다음 단계를 평가하기 위함이다.
2. 현재 `PrototypePack`은 train split으로부터 category centroid를 만드는 build artifact다.
3. validation split은 이후 scoring 품질, category separation, threshold calibration, 회귀 검증에 남겨둬야 한다.
4. 반면 균형화는 데이터 분포 자체를 바꾸는 작업이라 evaluation hygiene 문제와는 다른 층위다.
5. 초기 단계에서 무리하게 균형화를 걸면 실제 데이터 prior와 다른 centroid가 생길 수 있고, smoke 단계에서는 오히려 무엇이 개선됐는지 해석이 어려워진다.
6. 현재 split 스크립트는 label별 stratified split이므로 각 클래스가 train/validation 양쪽에 모두 남는다.

### 왜 현재 pack을 debug backend라고 분리해서 말해야 하는가

1. 현재 pack은 실제 `mxbai` 임베딩으로 만든 것이 아니다.
2. `scripts/seed_prototypes.py`는 `hash_debug`와 `transformers_mxbai` 두 backend를 지원한다.
3. 지금 생성된 pack manifest에는 `backend: hash_debug`가 기록돼 있다.
4. 따라서 이 pack은 계약 검증, centroid 생성 경로 검증, smoke/e2e 폐회로 검증에는 유효하지만 semantic 품질을 대표한다고 말하면 안 된다.

### 임베딩 모델을 먼저 내려받아 train split 임베딩을 계산하는 제안에 대한 판단

그 방향은 맞다. 다만 다음처럼 분리해서 가는 것이 좋다.

1. 실모델 임베딩은 train split에 대해 먼저 계산한다.
2. 그 train embedding으로 category centroid를 다시 빌드한다.
3. validation split은 pack 생성에 섞지 말고 평가 전용으로 남긴다.
4. 가능하면 per-example embedding cache를 build/eval artifact로 저장해서 prototype 재생성, threshold 실험, scoring 비교를 빠르게 돌린다.
5. 다만 그 embedding cache는 runtime 배포 객체인 `PrototypePack`에 넣지 말고 별도 build artifact로 관리한다.
6. 더 중요한 점은 seeding 경로와 agent runtime 경로의 embedding/preprocess 조건이 나중에 일치해야 train-serve skew가 줄어든다는 것이다.

추가 확인 결과:

1. `agent/src/infrastructure/model_adapters/embedding/mxbai.py`는 아직 `NotImplementedError` 상태다.
2. 따라서 지금 당장 실모델 prototype seeding은 가능하더라도, agent runtime까지 실사용 semantic path가 완전히 닫힌 상태는 아니다.
3. 다음 단계는 `transformers_mxbai` 기반 seeding 실험, embedding cache 도입 여부 결정, validation 평가 지표 정의, agent 쪽 실제 embedding adapter 구현 순으로 잡는 것이 자연스럽다.

## User

그렇게 진행하도록 하고 그 전에 data/processed 폴더에서 각 3개의 폴더는 역할이 각각 뭐야?

## Assistant

`data/processed` 아래 3개 폴더는 같은 데이터 파이프라인의 서로 다른 단계다.

1. `data/processed/labeled_query_sets`
   - 정제와 라벨 매핑이 끝난 canonical build/eval 입력 데이터셋이다.
   - 원본 CSV나 raw label을 바로 쓰지 않고, `mapped_label_4`까지 고정된 JSONL과 manifest를 둔다.
   - prototype seeding과 split의 입력이 된다.

2. `data/processed/splits`
   - `labeled_query_set`을 train/validation으로 나눈 evaluation hygiene용 산출물이다.
   - 여기의 `train`은 prototype build에 쓰고, `validation`은 이후 평가에 남긴다.
   - split manifest는 seed, ratio, label별 분포를 기록한다.

3. `data/processed/prototype_packs`
   - split된 train 데이터에서 임베딩과 centroid 집계를 거쳐 만든 runtime 배포 객체다.
   - agent scoring이 직접 사용하는 semantic 기준 산출물이며, 원문이나 개별 임베딩은 포함하지 않는다.
   - 현재 저장된 pack은 `hash_debug` backend 기반 debug/smoke artifact다.

즉 역할을 한 줄로 줄이면 아래다.

- `labeled_query_sets`: 기준 데이터셋
- `splits`: build/eval 경계 관리
- `prototype_packs`: 배포용 semantic 산출물

## User

prototype_packs이 이해가 안가는데? 모델에 아직 넣지도 않은 상태 아닌가? 어떻게 저런 결과가 나온거지?
그리고 split은 몇대 몇으로 한거지? 해당 코드는 어디있어

## Assistant

헷갈릴 만한 지점이 맞다. 현재 `prototype_pack`은 "학습된 모델"이 아니라 "카테고리별 대표 벡터 묶음"이기 때문에, 별도의 classifier 학습 없이도 생성된다.

근거는 아래와 같다.

1. 계약 문서에서 `PrototypePack`은 FL 모델 파라미터가 아니고, 카테고리별 대표 벡터를 담는 배포용 semantic 기준 객체라고 정의되어 있다.
2. `scripts/seed_prototypes.py`는 입력 JSONL을 `mapped_label_4` 기준으로 묶은 뒤, 각 category의 텍스트를 임베딩한다.
3. 현재는 `hash_debug` backend를 사용했기 때문에 외부 모델 추론 없이도 텍스트에서 deterministic pseudo-embedding을 만든다.
4. `shared/src/services/prototypes/prototype_pack_builder.py`는 카테고리별 임베딩 묶음의 평균을 내어 centroid를 만들고, 이것을 `PrototypePack`으로 감싼다.

즉 현재 생성된 결과는:

- "모델이 학습된 결과"가 아니라
- "train split 텍스트들을 category별로 임베딩해서 평균 centroid를 만든 결과"다.

현재 pack manifest 기준 사실:

1. 입력은 `data/processed/splits/ourafla_mental_health_text_classification.v1.train.jsonl`
2. backend는 `hash_debug`
3. `embedding_model_id`는 `hash_debug/mxbai_shape_proxy`

따라서 지금 나온 pack은 semantic smoke/debug artifact로 보면 맞고, 실제 pretrained embedding model 기반 품질 pack은 아직 아니다.

split 비율은 현재 `validation_ratio = 0.1`이라서 90:10이다.

구체적으로는:

1. split 스크립트 기본값이 `--validation-ratio 0.1`
2. 각 `mapped_label_4` bucket 안에서 shuffle 후 validation 개수를 먼저 떼는 stratified split
3. 현재 manifest 결과는 총 49,612개 중 train 44,651 / validation 4,961

코드 위치:

1. split 비율 인자 정의: `scripts/split_labeled_query_set.py`
2. stratified split 로직: `scripts/split_labeled_query_set.py`
3. 현재 실행 결과 manifest: `data/processed/splits/ourafla_mental_health_text_classification.v1.manifest.json`
4. prototype seeding 로직: `scripts/seed_prototypes.py`
5. centroid 생성 로직: `shared/src/services/prototypes/prototype_pack_builder.py`

## User

디버그용 모델 제거하고 하기로 한 mxbai_large모델 다운로드 후 train부분 넣어서 카테고리 만들자
그럼 해당 코드부터 만들어야 겠네
지금까지처럼 언제든지 모델 및 데이터는 변환 가능하도록 종속 시키지마 객체지향적 패턴 잘 지키면서 진행해

## Assistant

아래 방향으로 구현을 진행했다.

1. 임베딩 구현을 script 내부 함수 분기에서 빼고, 공통 네임스페이스 패키지 `tracemind_embedding`으로 분리했다.
2. `EmbeddingAdapter`, `EmbeddingAdapterSpec`, `EmbeddingAdapterFactory`를 추가해 backend별 구현을 팩토리로 선택하도록 만들었다.
3. `hash_debug`와 `mxbai_large` 어댑터를 같은 패턴으로 구성했다.
4. `agent` 쪽 `mxbai` 래퍼도 공통 구현을 바라보도록 바꿨다.
5. `scripts/seed_prototypes.py`는 더 이상 임베딩 세부 구현을 직접 가지지 않고 어댑터 객체에만 의존하도록 리팩터링했다.
6. `mxbai`는 `cache_dir`, `device`, `task_prefix`, `local_files_only`를 받도록 만들어 나중에 모델/실행환경/캐시 조건이 바뀌어도 덜 얽히게 했다.
7. category 단위 진행 로그를 script에서 출력하도록 추가했다.

검증 결과:

1. `PYTHONPATH=.pythonlibs python3.11 -m pytest agent/tests/unit/test_embedding_adapters.py agent/tests/unit/test_embedding_factory.py shared/tests/unit/test_prototype_pack_builder.py`
2. 위 테스트 9개는 모두 통과했다.

환경 준비:

1. 로컬 `.pythonlibs`에 `sentence-transformers`, `transformers`, `torch`, `pytest`를 설치했다.
2. `hf_cache/models--mixedbread-ai--mxbai-embed-large-v1` 아래에 모델 캐시가 생성되었다.

실행 시도:

1. `mxbai_large`로 full train split seeding을 시작했다.
2. 캐시 재사용을 위해 `--local-files-only` 옵션도 CLI에 추가했다.
3. 실제 full train seeding은 CPU 추론 시간이 길어 진행 중이었고, 최종 pack 파일이 생성되기 전 사용자 중단 요청이 들어왔다.

## User

종료시켜

## Assistant

진행 중이던 `seed_prototypes.py` 관련 프로세스 종료 신호를 보냈고, 현재 `data/processed/prototype_packs`에는 새 `proto_ourafla_mxbai_large_v1` 산출물이 생성되지 않은 상태임을 확인했다.
