# USB/CoMatch/Prototype Extension Plan

> 이어가기 프롬프트
>
> 작업 위치는 `/home/jmgjmg102/tracemind_server`다. 이 문서는
> `docs/ai_context_manifest.yaml`과 `docs/execution_index.md`에 entry를 추가하지
> 않는 보조 architecture plan이다. 먼저 `AGENTS.md`, `docs/AGENTS.md`,
> `docs/ai_context_manifest.yaml`, `docs/execution_index.md`를 읽되 두 index 파일은
> 수정하지 않는다. 이어서 `docs/architecture/target-method-runtime-structure.md`,
> `docs/architecture/strategy-surface-refactor-plan.md`,
> `docs/architecture/method-owned-runtime-refactor-plan.md`,
> `docs/contracts/algorithm_extension_guide.md`,
> `docs/contracts/strategy_addition_playbook.md`,
> `docs/contracts/fl_ssl_method_capability_matrix.md`, `methods/README.md`,
> `methods/ssl/NEW_METHOD.md`, `methods/federated_ssl/NEW_METHOD.md`,
> `methods/prototype/README.md`, `conf/README.md`를 기준으로 이어간다.
> 목표는 USB 계열 SSL, CoMatch, prototype 기반 scorer/evidence/update 확장을
> `methods/`와 `conf/` 중심으로 닫고, `scripts`, `agent`, `main_server`에
> method-specific 분기가 새지 않게 하는 것이다. CoMatch projection head는 public
> classifier/update family가 아니라 algorithm-local auxiliary trainable module로
> 다룬다.

## Status

이 문서는 구현 지시서가 아니라 확장 구조 계획이다. 최종 용어와 소유 경계는
`docs/architecture/target-method-runtime-structure.md`가 우선하고, 새 strategy 추가
절차는 `docs/contracts/strategy_addition_playbook.md`와
`docs/contracts/algorithm_extension_guide.md`가 우선한다.

이번 점검에서 확인한 현재 상태:

- `methods/ssl` registry와 descriptor convention은 FixMatch, FreeMatch,
  FlexMatch, AdaMatch 같은 logits 기반 USB objective를 작은 파일 추가로 받을 수
  있다.
- `QuerySslAlgorithm`은 `compute_step(model, labeled_batch, unlabeled_batch)` 중심이라
  feature projection, memory queue, teacher EMA, adversarial inner step이 필요한
  방법론에는 좁다.
- row에는 `text`, `aug_0`, `aug_1`가 있지만 현재 `usb_multiview` batch는 strong view
  하나만 노출한다.
- `TextEncoderWithLinearHead.extract_pooled_features()`가 있어 CoMatch projection 입력은
  만들 수 있지만, 이 surface가 Query SSL algorithm contract로 드러나 있지는 않다.
- prototype package는 build, scoring, thresholding, evidence, training input으로 이미
  나뉘어 있으므로 SSL objective와 한 축으로 합치면 안 된다.

## Scope

이 계획은 세 확장군을 함께 고려한다.

1. USB/SemiLearn 계열 Query SSL method:
   FixMatch, FreeMatch, SoftMatch, CoMatch, SimMatch, MixMatch, MeanTeacher, VAT 등.
2. Prototype 기반 method mechanism:
   prototype scorer, pseudo-label evidence, threshold/selection, training input,
   future `prototype_pack` update family.
3. FL SSL method-owned path:
   FedMatch 이후 FedLGMatch, `(FL)^2`처럼 local objective, server/round policy,
   peer/global state를 함께 소유하는 method.

비목표:

- `docs/ai_context_manifest.yaml`와 `docs/execution_index.md` 갱신.
- USB 전체를 한 번에 포팅.
- `prototype_pack` update family placeholder leaf 선생성.
- CoMatch 때문에 `shared`, `agent`, `main_server`를 선제 수정.
- public `teacher_provider` strategy axis 부활.

## Axis Classification

새 method를 하나의 `Algorithm` 객체로만 보지 않고 capability 요구사항으로 분해한다.

| 변화 축 | 대표 예 | Owner |
|---|---|---|
| SSL objective | FixMatch, FreeMatch, CoMatch, VAT | `methods/ssl/algorithms/*` |
| SSL hook/state | pseudo-label, mask, DA, threshold, queue | `methods/ssl/hooks/*` 또는 method-local |
| Query view surface | weak, weak/strong, weak/two-strong, mix batch | `methods/adaptation/query_text_views/*` |
| Model output surface | logits, pooled features, teacher logits | update family model + `methods/ssl` capability contract |
| Auxiliary trainable module | CoMatch/SimMatch projection head | algorithm-local module, trainer lifecycle |
| Prototype build | single, kmeans, dbscan | `methods/prototype/building/*` |
| Prototype scoring/evidence | similarity score, threshold, evidence | `methods/prototype/{scoring,thresholding,evidence,training_inputs}` |
| Update family | `peft_text_encoder`, future `prototype_pack` | `methods/adaptation/*`, `methods/prototype/*`, `shared` only if payload changes |
| FL method identity | FedMatch, FedLGMatch, `(FL)^2` | `methods/federated_ssl/<method>/` |
| Runtime capability | peer context, server step, artifact materializer | `methods/federated_ssl` capability + runtime adapter |

원칙:

- CoMatch projection head는 classifier head 대체가 아니다. classifier logits가
  supervised/unsupervised prediction의 main path이고 projection은 contrastive graph
  regularization용 auxiliary module이다.
- Prototype scorer는 SSL objective 이름이 아니다. Prototype을 SSL에 쓰려면 scorer,
  evidence, training input capability로 연결한다.
- Prototype이 client/server update payload가 되는 순간에만 `prototype_pack`
  update family와 shared contract 변경을 검토한다.

## Capability Gates

새 method를 구현하기 전에 아래 gate를 먼저 통과시킨다.

### Gate 1. Central Query SSL 여부

한 optimizer step 안에서 local loss를 계산하고 중앙 pooled/offline control로 비교할 수
있으면 `methods/ssl/algorithms/<method>/`에 둔다.

기본 추가 범위:

```text
methods/ssl/algorithms/<method>/
conf/strategy_axes/ssl_objective/consistency_method/<method>_usb_v1.yaml
tests/unit/test_methods_<method>.py
```

이 범위 밖 수정이 필요하면 method-specific 우회인지 reusable capability인지 먼저
판단한다.

### Gate 2. Missing Query SSL Capability

현재 없는 요구사항은 method 이름이 아니라 capability 이름으로 연다.

- `view_surface`: `weak_only`, `weak_strong`, `weak_two_strong`, `mix_batch`
- `model_forward_surface`: `logits_only`, `logits_and_pooled_features`,
  `teacher_logits`
- `algorithm_state_surface`: `stateless`, `dataset_state`, `feature_queue`,
  `probability_queue`, `distribution_ema`
- `optimizer_lifecycle`: `single_loss_step`, `auxiliary_trainable_module`,
  `post_optimizer_update`, `inner_adversarial_step`
- `teacher_state`: offline artifact teacher와 trainer-local EMA teacher를 분리

### Gate 3. Prototype Role

Prototype을 쓰는 요청은 먼저 역할을 고른다.

- reference artifact만 필요하면 `PrototypePack` contract와 `methods/prototype/building`.
- inference/training score가 필요하면 `methods/prototype/scoring`.
- pseudo-label 근거가 필요하면 `methods/prototype/evidence`.
- accepted candidate selection이 필요하면 `methods/prototype/thresholding`.
- weak/strong training row 변환이 필요하면 `methods/prototype/training_inputs`.
- FL update payload가 필요하면 future `prototype_pack` update family gate를 연다.

### Gate 4. FL Method-Owned 여부

server/round policy, peer context, global pseudo-label state, label exposure scenario가
method identity와 함께 움직이면 `methods/federated_ssl/<method>/`에서 시작한다.
`conf/strategy_axes/fssl_method/<method>.yaml`은 descriptor와 compatibility가 실제로
생긴 뒤에만 추가한다.

## Phase Plan

### Phase 0. Capability matrix 고정

- USB method를 capability 기준으로 분류한다.
- Prototype role matrix를 같은 문서나 code-adjacent README에 링크한다.
- CoMatch는 central Query SSL algorithm인지, FL method-owned method가 아닌지 먼저
  명시한다.
- USB upstream provenance 형식을 정한다: source repo, commit, path, preserved core,
  intentional deviations, unsupported features.

검증:

- 문서에 fixed/changed variables, dataset/split/seed, metric, output metadata를 적을
  위치가 있다.
- 중앙 SSL은 pooled/offline control이고 FL SSL non-IID가 main ranking이라는 rail을
  깨지 않는다.

### Phase 1. Query SSL descriptor를 capability-aware로 확장

- `QuerySslAlgorithmDescriptor`가 required view뿐 아니라 model output, state,
  lifecycle requirement를 표현하게 한다.
- 기존 algorithms는 기본 capability로 migration한다.
- compatibility validator가 unsupported method/runtime pair를 bootstrap 전에 실패시킨다.

검증:

- 기존 FixMatch/FreeMatch/FlexMatch/AdaMatch unit tests가 유지된다.
- unsupported feature method를 현재 runner에 올리면 명확한 error로 실패한다.

### Phase 2. Query view surface 확장

- 현재 `usb_multiview`는 compatibility로 유지한다.
- 새 `weak_two_strong` batch surface를 추가한다.
- batch key는 기존 flat style과 맞춰 `strong_0_input_ids`,
  `strong_1_input_ids` 같은 명시적 이름을 우선 검토한다.
- method-local `comatch_view.py`를 만들지 않는다.

검증:

- 기존 algorithms는 `strong_input_ids`만 소비해 계속 동작한다.
- CoMatch/SimMatch류 descriptor는 two-strong view가 없으면 view validation에서 실패한다.

### Phase 3. Model output surface와 auxiliary module lifecycle 추가

- logits-only `TextBatchClassifier`는 유지한다.
- pooled feature가 필요한 method를 위해 feature-returning classifier protocol을 명명한다.
- algorithm이 auxiliary trainable parameters를 제공할 수 있게 한다.
- optimizer와 gradient clipping은 model params + auxiliary params를 함께 다룬다.
- delta extraction은 update family model params만 대상으로 제한해 auxiliary projection
  head가 server payload에 섞이지 않게 한다.
- checkpoint/resume은 auxiliary module state와 algorithm queue state를 함께 저장한다.

검증:

- projection head는 학습된다.
- projection head는 PEFT/classifier delta에 포함되지 않는다.
- resume checkpoint roundtrip에서 projection head와 queue가 복원된다.

### Phase 4. CoMatch v1 구현

CoMatch는 새 capability 구조의 acceptance method다.

보존할 USB core:

- classifier logits 기반 supervised CE.
- weak probability, queue distribution alignment, memory smoothing.
- confidence mask와 strong_0 consistency loss.
- `Q = probs @ probs.T`, diagonal self-loop, `contrast_p_cutoff`, row normalize.
- strong_0/strong_1 projection embedding contrastive loss.

의도적으로 제외 가능한 v1 범위:

- distributed `concat_all_gather`.
- EPASS multi-projection.
- USB `AlgorithmBase` wrapper shape.
- public teacher selector.

파일 범위:

```text
methods/ssl/algorithms/comatch/
conf/strategy_axes/ssl_objective/consistency_method/comatch_usb_v1.yaml
tests/unit/test_methods_comatch.py
```

### Phase 5. Prototype integration gates

Prototype scorer/evidence를 SSL objective에 연결할 때는 `methods/ssl`이 prototype score
계산을 소유하지 않는다. `methods/ssl` algorithm은 required evidence/scorer capability만
요구하고, prototype 계산은 `methods/prototype` owner를 호출한다.

`prototype_pack`을 update family로 승격할 때만 추가 gate를 연다.

- shared payload/state contract가 기존 `PrototypePack` artifact로 충분한지 결정.
- `methods/prototype`에 update builder, materialization, aggregation projection,
  preflight owner를 둔다.
- `conf/strategy_axes/model_architecture/update_family/prototype_pack.yaml`은 runtime
  callable이 실제로 준비된 뒤에만 추가한다.
- runtime은 generic bridge를 실행하고 prototype method 이름으로 분기하지 않는다.

### Phase 6. FL method-owned 확장

FedLGMatch, `(FL)^2`, prototype-assisted FL SSL method는 `methods/federated_ssl/<method>/`
에서 시작한다.

- global/local pseudo-label state는 `round_state_exchange` 같은 capability로 표현한다.
- labels-at-server/client regime은 split/exposure condition과 method scenario를 분리한다.
- method-owned lower axis를 public override로 되살리지 않는다.

## Guardrails

- `shared`에는 method 이름, threshold default, `sigma/psi`, runtime fallback을 넣지 않는다.
- `conf` leaf는 조합과 파라미터만 소유한다. 수식과 기본 의미는 `methods` owner가
  소유한다.
- `scripts`, `agent`, `main_server`에 `comatch_*`, `simmatch_*`, `freematch_*`,
  `<method>_server_policy` 같은 method-name runtime 파일을 만들지 않는다.
- Hook은 처음에는 method-local에 두고, 두 개 이상 method에서 같은 의미로 안정되면
  `methods/ssl/hooks`로 승격한다.
- Prototype build/scoring/threshold/evidence/training-input/update-family를 한 축으로
  합치지 않는다.
- Placeholder config leaf를 만들지 않는다. descriptor, runtime callable, compatibility
  test가 준비된 뒤 leaf를 추가한다.
- Old `adapter_family_name`, `lora_classifier`, `diagonal_scale` vocabulary를 새 plan에
  되살리지 않는다.

## Verification

구현 단계별로 아래에서 변경 범위에 맞춰 고른다.

```bash
uv run pytest tests/unit/test_methods_<method>.py
uv run pytest tests/unit/test_peft_fixmatch_runner.py
uv run pytest tests/unit/test_scripts_hydra_configs.py
uv run pytest tests/architecture/test_layer_dependencies.py
uv run ruff check methods conf tests
```

새 update family를 열면 추가로 payload producer/consumer contract test, aggregation
projection test, Hydra compose test를 요구한다. FL method-owned path는 먼저 1-round
smoke, 필요 시 5-round reduced run으로 wiring과 metadata를 확인한 뒤 full-budget으로
올린다.

## Success Criteria

- 현재 seam 안에 들어오는 USB method는 method package, Hydra leaf, unit test만 추가한다.
- 새 capability가 필요할 때는 capability owner만 확장하고, 다음 method는 runner 수정 없이
  그 capability를 재사용한다.
- CoMatch projection head는 optimizer/checkpoint에 포함되지만 server update payload에는
  포함되지 않는다.
- Prototype은 SSL method identity가 아니라 scorer/evidence/update family role로
  명확히 분리된다.
- `docs/ai_context_manifest.yaml`과 `docs/execution_index.md`를 바꾸지 않아도 이 문서
  상단 프롬프트만으로 다음 채팅에서 이어갈 수 있다.
