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
이전 임시 문서 `QUERY_SSL_USB_EXTENSION_PLAN.md`의 유효한 구현 메모는 이 문서로
승격했고, 이후 판단은 이 문서만 기준으로 한다.

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

GitHub source audit 기준:

- `microsoft/Semi-supervised-learning`의 `semilearn/algorithms`는 `adamatch`,
  `comatch`, `crmatch`, `dash`, `defixmatch`, `fixmatch`, `flexmatch`,
  `freematch`, `meanteacher`, `mixmatch`, `multimatch`, `pimodel`,
  `pseudolabel`, `refixmatch`, `remixmatch`, `sequencematch`, `simmatch`,
  `softmatch`, `uda`, `vat` 등을 algorithm family로 둔다.
- 공통 구조는 `AlgorithmBase + registered algorithm + hook` 형태지만, TraceMind는
  이 class hierarchy를 그대로 복사하지 않는다. 필요한 수식과 state lifecycle만
  capability owner에 맞게 포팅한다.
- 반복되는 hook/capability는 pseudo-labeling, masking/weighting,
  distribution alignment, memory bank, projection head, mix batch,
  EMA teacher, adversarial perturbation, auxiliary task head다.
- CoMatch는 이 계획의 첫 acceptance method일 뿐 최종 목표가 아니다. CoMatch로
  projection/memory/two-strong capability를 검증한 뒤 SimMatch, SoftMatch,
  MixMatch/ReMixMatch, VAT/MeanTeacher 같은 방법론이 같은 구조를 재사용할 수 있어야
  한다.

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

### Trainable/Scorer Target Structure

USB/SemiLearn 확장은 `peft_text_encoder` 전용 구조로 가면 안 된다. 중앙 Query SSL과
미래 FL/runtime translation은 아래 축을 독립적으로 조합할 수 있어야 한다.

| 축 | 대표 값 | Owner |
|---|---|---|
| Trainable surface | `full_text_encoder`, `frozen_text_encoder_linear_head`, `peft_text_encoder` | `methods/adaptation/*`, `conf/strategy_axes/model_architecture/trainable_surface` |
| PEFT mechanism | `none`, `lora`, `dora`, `rslora` | `methods/adaptation/peft_adapters/*`, `conf/strategy_axes/model_architecture/peft` |
| Update family / trainable state | `linear_head`, `peft_text_encoder`, future `prototype_pack`, future full-model state | `methods/classification/*`, `methods/adaptation/*`, `methods/prototype/*`, `shared` only when payload changes |
| Scorer family | `classifier_logits`, `prototype_similarity` | `methods/classification/*`, `methods/prototype/scoring/*` |
| SSL objective | `fixmatch`, `freematch`, `softmatch`, `comatch`, `simmatch`, `mixmatch`, `vat` | `methods/ssl/algorithms/*` |

원하는 조합 예:

```text
trainable_surface=full_text_encoder
peft_adapter=none
scorer=classifier_logits
ssl_objective=fixmatch
```

```text
trainable_surface=frozen_text_encoder_linear_head
peft_adapter=none
update_family=linear_head
scorer=classifier_logits
ssl_objective=softmatch
```

```text
trainable_surface=peft_text_encoder
peft_adapter=lora
update_family=peft_text_encoder
scorer=classifier_logits
ssl_objective=comatch
```

```text
trainable_surface=peft_text_encoder
peft_adapter=dora
update_family=peft_text_encoder
scorer=classifier_logits
ssl_objective=simmatch
```

```text
scorer=prototype_similarity
prototype/build_strategy=kmeans
prototype_role=evidence
ssl_objective=<prototype-assisted method>
```

해석 규칙:

- `lora_classifier`, `dora_classifier` 같은 이름으로 PEFT mechanism과 classifier/scorer를
  다시 합치지 않는다. LoRA/DoRA는 mechanism이고 classifier/prototype은 scorer 또는
  update family다.
- `full_text_encoder`는 중앙 SSL/offline control에서는 trainable surface로 가능해야
  한다. FL update family로 승격하려면 full-model payload, aggregation, privacy
  contract gate를 별도로 연다.
- `frozen_text_encoder_linear_head`는 backbone은 고정하고 `linear_head`만 학습/공유하는
  구조다. 이것은 PEFT가 없는 정상 조합이다.
- `prototype_similarity`는 classifier head 대체 scorer가 될 수 있지만, prototype이
  client/server update payload가 되는 순간에는 `prototype_pack` update family와 shared
  contract gate를 따로 연다.
- SSL objective는 위 trainable/scorer 축을 직접 소유하지 않는다. Objective는 필요한
  view/model output/state capability를 요구하고, 실제 model/update/scorer 조립은 각
  owner가 맡는다.

원칙:

- CoMatch projection head는 classifier head 대체가 아니다. classifier logits가
  supervised/unsupervised prediction의 main path이고 projection은 contrastive graph
  regularization용 auxiliary module이다.
- Prototype scorer는 SSL objective 이름이 아니다. Prototype을 SSL에 쓰려면 scorer,
  evidence, training input capability로 연결한다.
- Prototype이 client/server update payload가 되는 순간에만 `prototype_pack`
  update family와 shared contract 변경을 검토한다.
- `QuerySslAlgorithmDescriptor`는 모든 축을 직접 field로 계속 흡수하는 class가
  아니다. Descriptor는 method identity, factory, required view, 작은 capability
  requirement value object만 드러낸다. Projection module, queue tensor, optimizer
  조립, checkpoint mutation, prototype score 계산은 각 owner module에 둔다.
- 새 축이 필요하면 먼저 owner를 정한다. Descriptor에 field를 추가하기 전에
  해당 값이 여러 method가 공유하는 capability requirement인지, 아니면
  method-local state/config인지 확인한다.

초기 descriptor shape는 아래 정도로 제한한다.

```python
QuerySslAlgorithmDescriptor(
    algorithm_name="comatch",
    display_name="CoMatch",
    required_views=QuerySslRequiredViews(...),
    runtime_requirements=QuerySslRuntimeRequirements(
        batch_surface="weak_strong_pair",
        model_outputs=frozenset({"logits", "pooled_features"}),
        optimizer_lifecycle=frozenset({"auxiliary_trainable_module"}),
        algorithm_state_surface=frozenset({"feature_queue", "probability_queue"}),
        step_context_required=True,
    ),
    algorithm_factory=...,
)
```

`QuerySslRuntimeRequirements`는 compatibility validation과 runner bootstrap 판단만
돕는다. Algorithm 수식이나 실행 default의 source of truth가 되면 안 된다.

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
- `input_transform_surface`: `none`, `mix_batch`, `manifold_mixup`,
  `auxiliary_task_view`
- `optimizer_lifecycle`: `single_loss_step`, `auxiliary_trainable_module`,
  `post_optimizer_update`, `inner_adversarial_step`, `bn_freeze_scope`
- `teacher_state`: offline artifact teacher와 trainer-local EMA teacher를 분리

중앙 Query SSL에서 처음 열 capability는 USB/SemiLearn 전체를 받는 이름으로 열되,
구현 범위는 CoMatch acceptance path에 필요한 만큼으로 제한한다.

```text
QuerySslStepContext
usb_weak_strong_pair view surface
logits_and_pooled_features model output surface
auxiliary_trainable_module optimizer lifecycle
feature/probability memory bank algorithm state
queue distribution alignment hook
```

이 이름들은 CoMatch 전용이 아니다. SimMatch는 projection/memory/queue DA를,
SoftMatch는 stateful EMA DA와 weighting을, MixMatch/ReMixMatch는 mix batch와
BN-freeze scope를, VAT는 inner adversarial step을, MeanTeacher는 trainer-local EMA
teacher를 같은 capability vocabulary 위에서 요구해야 한다.

CoMatch 뒤에 별도 gate로 열 후보:

| 다음 축 | 대표 method | 필요한 추가 capability |
|---|---|---|
| stateful weighting/DA | SoftMatch | weighting hook state export/load |
| projection memory bank | SimMatch | labeled index surface, feature bank, optional EMA teacher |
| mix batch | MixMatch, ReMixMatch | mixup input transform, BN-freeze scope |
| auxiliary task head | ReMixMatch, CRMatch | auxiliary trainable head, auxiliary view source |
| adversarial inner step | VAT | inner adversarial step, embedding perturbation surface |
| trainer-local EMA teacher | MeanTeacher, SimMatch | post-step EMA lifecycle |

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

- USB/SemiLearn method를 capability 기준으로 분류한다. CoMatch만이 아니라 최소
  아래 tier를 기준으로 정리한다.

| Tier | 대표 method | 먼저 필요한 owner |
|---|---|---|
| A. logits-only weak/strong | FixMatch, UDA, PseudoLabel | `methods/ssl/algorithms/*` |
| B. stateful threshold/DA/weighting | FlexMatch, FreeMatch, AdaMatch, SoftMatch | `methods/ssl/hooks/*`, algorithm state |
| C. projection + memory bank | CoMatch, SimMatch | model output/extension lifecycle, memory state |
| D. mix/multi-view transform | MixMatch, ReMixMatch, MultiMatch | query view/input transform surface |
| E. EMA/adversarial/custom loop | MeanTeacher, VAT, CRMatch | trainer lifecycle capability |

- Prototype role matrix를 같은 문서나 code-adjacent README에 링크한다.
- CoMatch는 central Query SSL algorithm인지, FL method-owned method가 아닌지 먼저
  명시하되, capability가 CoMatch 이름으로 닫히지 않게 한다.
- USB upstream provenance 형식을 정한다: source repo, commit, path, preserved core,
  intentional deviations, unsupported features.
- CoMatch 기준 원본은 `microsoft/Semi-supervised-learning` commit
  `1ef4cbebcc0b368158315aeb425053858cf6c845`의
  `semilearn/algorithms/comatch/comatch.py`다.

검증:

- 문서에 fixed/changed variables, dataset/split/seed, metric, output metadata를 적을
  위치가 있다.
- 중앙 SSL은 pooled/offline control이고 FL SSL non-IID가 main ranking이라는 rail을
  깨지 않는다.

### Phase 1. Query SSL descriptor를 capability-aware로 확장

- `QuerySslAlgorithmDescriptor`가 required view뿐 아니라 model output, state,
  lifecycle requirement를 작은 `QuerySslRuntimeRequirements` 값으로 표현하게 한다.
- `QuerySslStepContext`를 추가한다.

```python
@dataclass(frozen=True, slots=True)
class QuerySslStepContext:
    epoch_index: int
    step_index: int
    global_step: int
    total_train_steps: int
    num_classes: int
    device: torch.device
```

- 기존 algorithm은 context 없이 계속 동작하게 compatibility helper를 둔다. 새
  context-aware method만 context를 소비한다.
- 기존 algorithms는 기본 capability로 migration한다.
- compatibility validator가 unsupported method/runtime pair를 bootstrap 전에 실패시킨다.

검증:

- 기존 FixMatch/FreeMatch/FlexMatch/AdaMatch unit tests가 유지된다.
- unsupported feature method를 현재 runner에 올리면 명확한 error로 실패한다.
- `scripts/support/query_ssl_peft/runners/consistency.py`는 method 이름으로 분기하지 않고
  descriptor capability만 읽는다.

### Phase 2. Query view surface 확장

- 현재 `usb_multiview`는 compatibility로 유지한다.
- 새 `usb_weak_strong_pair` batch surface를 추가한다.
- batch key는 기존 flat style과 맞춰 `strong_0_input_ids`,
  `strong_1_input_ids` 같은 명시적 이름을 우선 검토한다.
- method-local `comatch_view.py`를 만들지 않는다.

새 batch surface:

```text
weak_input_ids
weak_attention_mask
strong_0_input_ids
strong_0_attention_mask
strong_1_input_ids
strong_1_attention_mask
```

검증:

- 기존 algorithms는 `strong_input_ids`만 소비해 계속 동작한다.
- CoMatch/SimMatch류 descriptor는 two-strong view가 없으면 view validation에서 실패한다.

### Phase 3. Model output surface와 auxiliary module lifecycle 추가

- logits-only `TextBatchClassifier`는 유지한다.
- pooled feature가 필요한 method를 위해 feature-returning classifier protocol을 명명한다.
- algorithm이 auxiliary trainable parameters를 제공할 수 있게 한다. 초기 후보 위치는
  `methods/ssl/model_capabilities.py`와
  `methods/adaptation/peft_text_encoder/training/ssl_model_extensions.py`다.
- optimizer와 gradient clipping은 model params + auxiliary params를 함께 다룬다.
- delta extraction은 update family model params만 대상으로 제한해 auxiliary projection
  head가 server payload에 섞이지 않게 한다.
- checkpoint/resume은 auxiliary module state와 algorithm queue state를 함께 저장한다.

검증:

- projection head는 학습된다.
- projection head는 PEFT/classifier delta에 포함되지 않는다.
- resume checkpoint roundtrip에서 projection head와 queue가 복원된다.

### Phase 4. CoMatch v1 구현

CoMatch는 새 capability 구조의 첫 acceptance method다. 목표는 CoMatch 하나를
끼우는 것이 아니라 projection/memory/two-strong capability를 열어 SimMatch류도
같은 구조를 탈 수 있게 하는 것이다.

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
  README.md
  comatch.py
  memory_bank.py
  original_spec.py
conf/strategy_axes/ssl_objective/consistency_method/comatch_usb_v1.yaml
tests/unit/test_methods_comatch.py
```

CoMatch config는 USB parity와 ablation을 구분한다. USB parity preset에는
`temperature`, `p_cutoff`, `contrast_p_cutoff`, `queue_batch`, `smoothing_alpha`,
`da_len`, `proj_size`, `lambda_u`, `lambda_c`, `supervised_loss_weight`,
`unlabeled_batch_size`, `require_multiview`를 둔다.

CoMatch 직후 구조 generality 검증 후보:

- `SoftMatch`: 새 projection 없이 stateful weighting/DA 추가가 descriptor와 state
  lifecycle로 닫히는지 확인한다.
- `SimMatch`: CoMatch가 연 projection/memory/queue DA surface를 재사용할 수 있는지
  확인한다. 단, labeled index bank나 EMA teacher 요구가 생기면 별도 capability로 연다.
- `MixMatch`: mix batch/input transform capability가 필요한지 확인하되, CoMatch 구현
  중에는 선제 구현하지 않는다.

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

## Detailed Execution Sequence

아래 순서가 실제 작업 계획이다. 각 step은 직전 step의 테스트가 통과한 뒤에만 다음으로
넘어간다. CoMatch는 첫 acceptance method지만, capability 이름과 파일 경계는
USB/SemiLearn 계열 전체를 받는 형태로 유지한다.

### Step 1. Capability matrix와 descriptor guard 고정

상태: 완료 (2026-06-01)

목표:

- `QuerySslAlgorithmDescriptor`가 method identity와 requirement만 표현하게 한다.
- capability requirement가 method 이름으로 새지 않게 한다.
- 기존 FixMatch/FreeMatch/FlexMatch/AdaMatch/PseudoLabel descriptor가 기본 capability를
  명시하게 한다.

예상 변경:

```text
methods/ssl/base.py
tests/unit/test_methods_fixmatch.py
tests/unit/test_methods_freematch.py
tests/unit/test_methods_flexmatch.py
tests/unit/test_methods_adamatch.py
tests/unit/test_methods_pseudolabel.py
```

예상 코드 표면:

```python
@dataclass(frozen=True, slots=True)
class QuerySslRuntimeRequirements:
    batch_surface: str = "weak_strong"
    model_outputs: frozenset[str] = frozenset({"logits"})
    algorithm_state_surface: frozenset[str] = frozenset({"stateless"})
    input_transform_surface: str = "none"
    optimizer_lifecycle: frozenset[str] = frozenset({"single_loss_step"})
    teacher_state: str = "none"
    step_context_required: bool = False
```

검증:

```bash
uv run pytest tests/unit/test_methods_fixmatch.py tests/unit/test_methods_freematch.py \
  tests/unit/test_methods_flexmatch.py tests/unit/test_methods_adamatch.py \
  tests/unit/test_methods_pseudolabel.py
```

완료 기준:

- 기존 method behavior가 바뀌지 않는다.
- descriptor field가 tensor, module, optimizer object를 들지 않는다.

완료 기록:

- `QuerySslRuntimeRequirements`를 descriptor의 작은 capability value object로 추가했다.
- runtime requirement 값은 닫힌 enum이 아니다. 코드에는 현재 built-in algorithm이
  실제로 쓰는 기본값만 상수화하고, 미래 capability는 실제 producer/consumer/test가 생기는
  단계에서 추가한다.
- FixMatch/FreeMatch/FlexMatch/AdaMatch/PseudoLabel descriptor가 현재 batch, model
  output, state, transform, optimizer, teacher requirement를 노출한다.
- `tests/unit/test_query_ssl_runtime_requirements.py`가 descriptor가 implementation object,
  tensor, optimizer object 대신 문자열/frozenset requirement만 갖는 경계를 검증한다.

### Step 2. Step context와 compatibility call helper 추가

상태: 완료 (2026-06-01)

목표:

- step-dependent policy가 `configure_training`/`configure_dataset`에 억지로 숨지 않게 한다.
- 기존 algorithm은 context를 몰라도 그대로 실행된다.

예상 변경:

```text
methods/ssl/base.py
methods/adaptation/peft_text_encoder/training/loops.py
tests/unit/test_peft_encoder_training_core.py
```

예상 코드 표면:

```python
QuerySslStepContext(
    epoch_index=epoch,
    step_index=step_index,
    global_step=completed_steps + 1,
    total_train_steps=step_budget.total_train_steps,
    num_classes=len(categories),
    device=torch.device(device),
)
```

`loops.py`는 algorithm별 `if algorithm_name == ...`를 넣지 않고 helper를 통해
context-aware algorithm에만 context를 넘긴다.

검증:

```bash
uv run pytest tests/unit/test_peft_encoder_training_core.py
```

완료 기준:

- 기존 `_CountingQuerySslAlgorithm`처럼 context 없는 test double이 계속 동작한다.
- context 필요한 fake algorithm이 global step과 total step을 검증할 수 있다.

완료 기록:

- `QuerySslStepContext`와 `compute_query_ssl_algorithm_step(...)` helper를 추가했다.
- `train_query_ssl_classifier(...)`는 method 이름으로 분기하지 않고 helper를 통해
  context-aware algorithm에만 context를 전달한다.
- 기존 context 없는 algorithm은 `compute_step(...)` 경로로 계속 실행된다.
- `tests/unit/test_peft_encoder_training_core.py`가 context-aware fake algorithm으로
  `global_step`, `total_train_steps`, `num_classes`, `device` 전달을 검증한다.

### Step 3. Query view surface를 weak/strong pair까지 확장

상태: 완료 (2026-06-01)

목표:

- `usb_multiview`는 기존 compatibility로 유지한다.
- 새 `usb_weak_strong_pair`는 `aug_0`, `aug_1`을 모두 batch에 노출한다.
- view 선택은 `methods/adaptation/query_text_views`가 소유하고 scripts는 descriptor의
  `view_builder_name`만 전달한다.

예상 변경:

```text
methods/adaptation/query_text_views/view_rows.py
methods/adaptation/query_text_views/data.py
methods/adaptation/query_text_views/query_ssl_views.py
methods/adaptation/query_text_views/unlabeled_preparation.py
tests/unit/test_query_text_views_data.py
tests/architecture/test_layer_dependencies.py
```

예상 batch:

```text
query_ids
row_indices
weak_input_ids
weak_attention_mask
strong_0_input_ids
strong_0_attention_mask
strong_1_input_ids
strong_1_attention_mask
```

검증:

```bash
uv run pytest tests/unit/test_query_text_views_data.py \
  tests/architecture/test_layer_dependencies.py
```

완료 기준:

- 기존 FixMatch류는 `strong_input_ids` batch를 계속 받는다.
- CoMatch/SimMatch류 descriptor가 `usb_weak_strong_pair`를 요구할 수 있다.

완료 기록:

- `USB_WEAK_STRONG_PAIR_BUILDER_NAME = "usb_weak_strong_pair"` view builder를
  `methods/adaptation/query_text_views` owner 안에 추가했다.
- 새 `TextWeakStrongPairDataset`과 `build_weak_strong_pair_dataloader(...)`는 strict
  USB `text/aug_0/aug_1` row만 받아 `weak_*`, `strong_0_*`, `strong_1_*` batch key를
  만든다.
- 기존 `usb_multiview`는 legacy `weak_text/strong_text` compatibility와
  `strong_input_ids` batch를 그대로 유지한다.
- augmentation preparation은 `usb_multiview`와 `usb_weak_strong_pair` 모두 strict USB
  candidate preparation을 재사용한다.
- `tests/architecture/test_layer_dependencies.py`가 script adapter에
  `usb_weak_strong_pair` method/view 분기가 새지 않도록 guard한다.

### Step 4. Model output capability와 auxiliary module lifecycle 추가

상태: 완료 (2026-06-01)

목표:

- logits-only model protocol은 유지한다.
- pooled feature가 필요한 method가 명시적으로 `logits_and_pooled_features`를 요구한다.
- projection head 같은 auxiliary module은 optimizer, grad clipping, checkpoint에 포함되지만
  PEFT/classifier server update payload에는 포함되지 않는다.

예상 변경:

```text
methods/ssl/model_capabilities.py
methods/adaptation/peft_text_encoder/training/ssl_model_extensions.py
methods/adaptation/peft_text_encoder/training/loops.py
methods/adaptation/common/query_ssl_training_resume.py
tests/unit/test_peft_encoder_training_core.py
```

예상 책임:

- `methods/ssl/model_capabilities.py`: protocol과 capability 이름.
- `ssl_model_extensions.py`: PEFT text encoder trainer가 auxiliary module을 붙이고,
  trainable parameters/checkpoint payload를 만든다.
- Algorithm package: 어떤 auxiliary module이 필요한지와 수식 의미.

검증:

```bash
uv run pytest tests/unit/test_peft_encoder_training_core.py
```

완료 기준:

- projection head parameter가 optimizer step으로 변한다.
- checkpoint roundtrip 후 auxiliary module state가 복원된다.
- update payload extraction test에서 projection head가 payload에 섞이지 않는다.

완료 기록:

- `methods/ssl/model_capabilities.py`에 pooled feature classifier capability와
  algorithm-local auxiliary module provider 계약을 추가했다.
- `methods/adaptation/peft_text_encoder/training/ssl_model_extensions.py`가 PEFT Query SSL
  trainer에서 auxiliary module을 device, optimizer parameter, train/eval lifecycle에
  연결한다.
- `train_query_ssl_classifier(...)`는 model parameter와 auxiliary parameter를 함께
  optimizer/gradient clipping 대상으로 다루되, FedProx 기준 snapshot은 기존 update family
  model parameter에만 적용한다.
- Query SSL resume checkpoint가 `auxiliary_module_state_dicts`를 저장/복원한다.
- auxiliary module은 PEFT model submodule으로 등록하지 않는다. 따라서 기존
  PEFT/classifier delta extraction과 server update payload에는 projection head가 섞이지
  않는다.
- `tests/unit/test_query_ssl_model_capabilities.py`와
  `tests/unit/test_peft_encoder_training_core.py`가 pooled feature capability, auxiliary
  parameter update, checkpoint roundtrip, payload 분리를 검증한다.

### Step 5. Queue DA와 memory bank primitive 추가

상태: 완료 (2026-06-01)

목표:

- SemiLearn의 `DistAlignEMAHook`/`DistAlignQueueHook` 차이를 TraceMind hook surface로
  옮긴다.
- CoMatch/SimMatch류 memory bank는 method-local primitive에서 시작한다.
- 두 개 이상 method에서 같은 의미로 안정되면 공통 hook/helper로 승격한다.

예상 변경:

```text
methods/ssl/hooks/distribution_alignment.py
methods/ssl/algorithms/comatch/memory_bank.py
tests/unit/test_methods_ssl_hooks.py
tests/unit/test_methods_comatch.py
```

검증:

```bash
uv run pytest tests/unit/test_methods_ssl_hooks.py tests/unit/test_methods_comatch.py
```

완료 기준:

- queue DA state가 export/load 가능하다.
- memory bank pointer, feature queue, probability queue가 deterministic unit test로
  검증된다.

완료 기록:

- `methods/ssl/hooks/distribution_alignment.py`에 USB `DistAlignQueueHook` 의미를
  옮긴 `QueueDistributionAlignmentHook`을 추가했다.
- queue DA는 batch probability 평균을 `p_model` queue에 저장하고,
  `p_target.mean() / p_model.mean()`으로 unlabeled probability를 정렬한다.
- queue DA `export_state()`/`load_state(...)`가 `p_model`, `p_model_ptr`, `p_target`,
  `p_target_ptr`를 checkpoint-safe tensor state로 roundtrip한다.
- `methods/ssl/algorithms/comatch/memory_bank.py`를 CoMatch method-local primitive로
  추가했다. 아직 `comatch` algorithm registry entry는 열지 않는다.
- `CoMatchMemoryBank`가 feature/probability queue, ring-buffer pointer, memory smoothing,
  state export/load를 소유한다.
- registry builtin loader는 method package에 같은 이름 entrypoint가 없으면 건너뛴다.
  따라서 `comatch/memory_bank.py` 같은 method-local primitive를 먼저 둘 수 있고,
  실제 `comatch.py` 등록은 Step 6에서 한다.
- `tests/unit/test_methods_ssl_hooks.py`와 `tests/unit/test_methods_comatch.py`가 queue DA
  state, memory bank pointer/wrap, probability smoothing, state roundtrip을 검증한다.

### Step 6. CoMatch tensor core와 algorithm adapter 구현

목표:

- CoMatch 수식 core를 tensor-level function으로 먼저 검증한다.
- Algorithm adapter는 batch/model/state wiring만 맡는다.
- 원본 USB와 의도적 차이를 `original_spec.py`나 README에 남긴다.

예상 변경:

```text
methods/ssl/algorithms/comatch/
  README.md
  comatch.py
  memory_bank.py
  original_spec.py
conf/strategy_axes/ssl_objective/consistency_method/comatch_usb_v1.yaml
tests/unit/test_methods_comatch.py
tests/unit/test_scripts_hydra_configs.py
```

검증:

```bash
uv run pytest tests/unit/test_methods_comatch.py tests/unit/test_scripts_hydra_configs.py
```

완료 기준:

- supervised CE, weak probability, queue DA, memory smoothing, confidence mask,
  strong consistency, graph contrastive loss가 각각 unit test로 보인다.
- Hydra leaf는 parameter와 조합만 소유한다.

### Step 7. Runner smoke와 architecture guard 고정

목표:

- scripts runner가 method 이름으로 분기하지 않음을 test로 고정한다.
- descriptor capability만으로 view/model/lifecycle bootstrap이 결정된다.

예상 변경:

```text
tests/unit/test_peft_fixmatch_runner.py
tests/architecture/test_layer_dependencies.py
```

검증:

```bash
uv run pytest tests/unit/test_peft_fixmatch_runner.py \
  tests/architecture/test_layer_dependencies.py
```

완료 기준:

- `scripts/support/query_ssl_peft/runners/consistency.py`에 `comatch`, `simmatch`,
  `softmatch` 같은 method-name branch가 없다.
- unsupported capability 조합은 bootstrap 전에 명확한 error로 실패한다.

### Step 8. Generality probe method 선택

목표:

- CoMatch 구현 직후 구조가 CoMatch 전용이 아닌지 검증한다.
- 다음 method를 하나만 골라 capability 재사용성을 확인한다.

권장 순서:

1. `SoftMatch`: 새 projection 없이 stateful weighting/DA가 닫히는지 확인한다.
2. `SimMatch`: CoMatch projection/memory/queue DA surface 재사용성을 확인한다.
3. `MixMatch`: mix batch/input transform seam이 필요한 시점에 연다.

완료 기준:

- 두 번째 method 추가가 `methods/ssl/algorithms/<method>/`, Hydra leaf, unit test 중심으로
  닫힌다.
- 새 capability가 필요하면 그 owner만 확장하고 runner method branch는 만들지 않는다.

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
- `QuerySslAlgorithmDescriptor`에 concrete implementation object, tensor state,
  optimizer parameter group, checkpoint serialization rule을 넣지 않는다.
- `methods/ssl/model_capabilities.py`는 protocol/capability name만 소유한다. PEFT text
  encoder에 실제 auxiliary module을 붙이는 실행 lifecycle은
  `methods/adaptation/peft_text_encoder/training/`이 소유한다.
- `methods/ssl/algorithms/comatch/`의 projection head는 CoMatch/SimMatch류 auxiliary
  module이지 `linear_head`나 `prototype_pack` update family가 아니다.
- capability 이름에 method 이름을 넣지 않는다. `comatch_projection_head`,
  `simmatch_memory_bank`, `mixmatch_runner`가 아니라
  `auxiliary_trainable_module`, `feature_memory_bank`, `mix_batch_transform` 같은 이름을 쓴다.

## Verification

구현 단계별로 아래에서 변경 범위에 맞춰 고른다.

```bash
uv run pytest tests/unit/test_methods_<method>.py
uv run pytest tests/unit/test_query_text_views_data.py
uv run pytest tests/unit/test_peft_encoder_training_core.py
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
