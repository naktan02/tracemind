# Prototype Scripts

이 디렉터리는 prototype pack 생성, 평가, build-state 갱신 같은
prototype artifact 작업 entrypoint를 모은다.

중요:

- 여기는 `prototype artifact lifecycle` 레일이다.
- 즉 `seed / evaluate / pull / activate / report`처럼 이미 정해진
  prototype pack을 만들고 검증하고 배포 상태를 다루는 workflow가 주된 역할이다.
- `어떤 prototype 전략이 더 좋은지`를 비교하는 연구형 실험은
  `scripts/experiments/prototype_analysis/`가 맡는다.
- 둘 다 prototype을 다루지만, 이 폴더는 `artifact 작업`, experiments는
  `비교 실험`이라는 점이 경계다.

## 먼저 읽을 파일

1. `seed_prototypes.py`
   - dataset row로부터 최초 prototype pack/build state를 생성
2. `evaluate_prototype_pack.py`
   - 이미 만든 prototype pack을 validation/test set으로 평가
3. `seeding.py`
   - embedding adapter 생성과 rebuild 호출을 실제로 수행

평가 metric과 prototype projection 계산 core는 `methods/prototype/`가 소유한다.
이 폴더의 entrypoint는 artifact load/write, runtime adapter 호출, 표 출력만 맡는다.

## 바로 조절 가능한 축

- `embedding`
- `embedding.model_id`
- `embedding.revision`
- `runtime.device`
- `prototype_builder`
- `translation_model_id`
- `translation_model_revision`
- `translation_direction`

예시:

```bash
python -m scripts.workflows.prototype_pack.seed_prototypes \
  execution_context/embedding_adapter=hash_debug \
  strategy_axes/prototype/build_strategy=kmeans \
  prototype_builder.candidate_ks=[2,3] \
  translation_model_id=facebook/nllb-200-distilled-600M \
  translation_model_revision=main \
  translation_direction=kor_Hang->eng_Latn
```

## 중요한 구분

- `embedding`은 실제 runtime knob다.
  `conf/execution_context/embedding_adapter/*.yaml`에서 backend/model을 고르고,
  `EmbeddingAdapterFactory`가 실제 adapter를 만든다.
- `translation_model_id`, `translation_model_revision`, `translation_direction`은
  현재 prototype pack/build state에 남는 provenance metadata다.
  이 값들이 곧바로 translation adapter를 instantiate해서
  원문을 번역한 뒤 임베딩한다는 뜻은 아니다.
- 실제 번역 모델 교체가 runtime knob로 열려 있는 경로는 현재
  `agent` inference pipeline 쪽이다.

## newcomer 메모

- embedding backend를 더 추가하고 싶으면
  [agent/src/infrastructure/model_adapters/embedding/factory.py](../../../agent/src/infrastructure/model_adapters/embedding/factory.py)를
  먼저 본다.
- prototype pack의 translation 관련 필드 의미는
  [docs/contracts/prototype_pack_v1.md](../../../docs/contracts/prototype_pack_v1.md)를
  같이 보는 편이 빠르다.
- 현재 전략 축 전체 상태는 `conf/README.md`와 관련 `conf/strategy_axes/**`
  leaf를 기준으로 본다.
