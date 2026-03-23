# 2026-03-23 Session Transcript

## User

`$tracemind-project 시작전에 궁금한것: 중앙서버와 로컬에이전트를 레포 따로 해서 진행하나? 아니면 나중에 분리하나?`

## Assistant

`지금 단계에서는 레포를 따로 나누지 말고 하나의 모노레포로 시작하는 게 맞습니다.

이유는 현재 계획 자체가 공유 계약 정의 -> 로컬 analytics MVP -> 중앙 normative server -> FL 확장 순서로 잡혀 있기 때문입니다. 즉 지금 가장 중요한 건 중앙/로컬을 조직적으로 분리하는 것보다, 둘이 공유하는 도메인과 계약을 안정적으로 만드는 것입니다. 이 단계에서 레포를 먼저 쪼개면 오히려 WindowSummary, NormPack, TrainingUpdateEnvelope 같은 핵심 계약이 양쪽에 중복되거나 동기화 비용이 생깁니다.

내 추천은 이렇습니다.

1. 지금은 한 레포로 간다.
2. 대신 코드 구조는 처음부터 논리적으로 분리한다.
3. 나중에 계약과 배포 경계가 안정되면 물리적으로 분리할지 판단한다.

구체적으로는 한 레포 안에서 이렇게 가는 게 좋습니다.

- src/api/local
- src/api/central
- src/core/domain
- src/core/ports
- src/services/local
- src/services/central
- src/infrastructure

이렇게 하면 지금은 개발이 빠르고, 나중에 정말 필요하면 local-agent와 central-server를 별도 레포로 떼어내기도 쉽습니다.

언제 분리하면 되냐면, 아래 조건이 생길 때입니다.

1. 중앙 서버와 로컬 에이전트의 배포 주기가 달라질 때
2. 팀이 나뉘어 각자 독립적으로 작업할 때
3. 공통 계약이 거의 안정되어 별도 패키지로 빼도 흔들리지 않을 때
4. CI, 테스트, 릴리스 파이프라인을 독립 운영할 필요가 생길 때

즉 지금은 모노레포 + 내부 모듈 분리가 가장 좋고, 나중에는 필요하면 이렇게 갈 수 있습니다.

- tracemind-core-contracts
- tracemind-local-agent
- tracemind-central-server

중요한 건 지금부터 분리 가능한 구조로 짜되, 레포는 아직 하나로 유지하는 것입니다.
그게 현재 단계에서 가장 미래지향적이면서도 수정 비용이 낮은 선택입니다.

원하면 다음으로 바로 모노레포 기준 최종 폴더 구조안을 확정해서 문서에 적어드리겠습니다.`

## User

`각 폴더가 어떤 기능을 할 예정인건지 모든 폴더 설명해줘`

## Assistant

`기준은 FL 확장까지 포함한 최신 구조를 기준으로 설명하겠습니다. 이 구조는 지금 MVP와 나중 parameter FL을 한 뼈대 안에 같이 수용하려는 목적입니다.

전체 원칙
- api: 바깥에서 들어오는 요청을 받는 입구
- core: 가장 오래 유지되는 도메인 규칙과 계약
- services: 실제 유스케이스를 조합하는 계층
- infrastructure: DB, HTTP, 모델, privacy 구현체 같은 교체 가능한 바깥 기술
- scripts: 수동 실행용 보조 도구
- tests: 검증 코드

src/
- 애플리케이션 런타임 코드 전체가 들어가는 루트입니다.
- 로컬 에이전트와 중앙 서버를 한 레포에서 운영하되, 내부 책임은 분리합니다.

src/api/
- FastAPI 같은 프레임워크 진입점만 둡니다.
- 여기서는 입력 검증, 요청 라우팅, 응답 변환만 하고 비즈니스 로직은 넣지 않습니다.

src/api/central/
- 중앙 서버용 API 진입점입니다.
- cohort 집계, 정책 배포, FL 라운드 관리 같은 중앙 책임만 노출합니다.

src/api/central/main.py
- 중앙 서버 앱 생성, 라우터 등록, 공통 미들웨어 설정 담당입니다.

src/api/central/routers/
- 중앙 서버 엔드포인트 파일 모음입니다.
- 기능별로 라우터를 나눠서 중앙 책임을 명확히 분리합니다.

src/api/central/routers/health.py
- 헬스체크, readiness, liveness 같은 운영 확인용 엔드포인트입니다.

src/api/central/routers/sync.py
- 로컬이 WindowSummary를 업로드하는 동기화 API입니다.
- analytics MVP에서 가장 먼저 실제로 쓰일 중앙 API입니다.

src/api/central/routers/norms.py
- 중앙이 계산한 NormPack을 내려주는 API입니다.
- 로컬이 peer norm을 가져갈 때 사용합니다.

src/api/central/routers/policies.py
- 전역 정책, 임계값, 버전 정책, cohort 최소 샘플 수 같은 설정 배포용입니다.

src/api/central/routers/agents.py
- 로컬 에이전트 등록, heartbeat, capability 보고용입니다.
- FL-ready 단계 이후 중요해집니다.

src/api/central/routers/fl_rounds.py
- 파라미터 연합학습 라운드 관리 API입니다.
- join, config, status, updates 흐름을 다룹니다.

src/api/central/routers/models.py
- 모델 manifest, 버전 메타데이터, 배포 대상 모델 조회용입니다.
- 로컬이 어떤 모델/헤드를 내려받아야 하는지 알게 합니다.

src/api/local/
- 로컬 에이전트용 API 진입점입니다.
- 브라우저 확장, 앱, 백그라운드 프로세스가 로컬 분석 기능을 호출하는 입구입니다.

src/api/local/main.py
- 로컬 앱 생성, 로컬 라우터 등록, 로컬 설정 로딩 담당입니다.

src/api/local/routers/
- 로컬 에이전트가 제공하는 엔드포인트들입니다.

src/api/local/routers/ingest.py
- 검색어, 메시지, 이벤트 같은 원본 입력을 받는 엔드포인트입니다.
- 여기서 받은 데이터는 이후 전처리/임베딩/요약으로 흐릅니다.

src/api/local/routers/assessment.py
- 현재 로컬 판단 결과, 최근 기준선 변화, 추천 리소스 결과를 조회하는 엔드포인트입니다.

src/api/local/routers/sync.py
- 중앙 서버와 통신을 트리거하는 로컬 엔드포인트입니다.
- summary 업로드, norm 동기화, 재시도 같은 흐름과 연결됩니다.

src/api/local/routers/training.py
- 나중에 로컬 학습 작업을 시작하거나 상태를 확인하는 엔드포인트입니다.
- parameter FL 단계 이후 의미가 커집니다.

src/core/
- 프로젝트에서 가장 중요한 계층입니다.
- 로컬/중앙/프레임워크와 무관하게 유지되는 규칙과 공통 개념을 둡니다.

src/core/domain/
- 비즈니스 도메인 그 자체입니다.
- TraceMind가 무엇을 다루는 시스템인지 정의하는 곳입니다.

src/core/domain/entities/
- 수명과 상태를 가진 핵심 객체들입니다.
- 예: QueryEvent, WindowSummary, NormPack, TrainingTask

src/core/domain/entities/query_event.py
- 로컬에서 수집한 원본 이벤트 모델입니다.
- timestamp, source, locale, normalized_text 같은 속성을 갖습니다.

src/core/domain/entities/scored_event.py
- 임베딩과 카테고리 점수 계산이 끝난 이벤트입니다.
- 원본 이벤트와 scoring 결과를 연결합니다.

src/core/domain/entities/window_summary.py
- 일정 기간의 요약 통계입니다.
- 중앙 서버로 보내는 최소 단위라서 매우 중요합니다.

src/core/domain/entities/baseline_profile.py
- 개인 기준선 상태입니다.
- 평균, 분산, slope, persistence, warm-up 여부를 담습니다.

src/core/domain/entities/norm_pack.py
- 중앙이 계산한 cohort 기준입니다.
- 로컬 판단에 쓰는 집단 기준 패키지입니다.

src/core/domain/entities/assessment_result.py
- 최종 로컬 판단 결과입니다.
- self shift, peer deviation, persistence, decision level 같은 값이 들어갑니다.

src/core/domain/entities/training_task.py
- 중앙이 로컬에 배포하는 학습 작업 정의입니다.
- 어떤 모델, 어떤 round, 어떤 데이터 범위를 쓸지 정의합니다.

src/core/domain/entities/training_update.py
- 로컬이 중앙으로 보내는 파라미터 업데이트 envelope입니다.
- 나중에 secure aggregation, DP, clipping 메타데이터도 연결될 수 있습니다.

src/core/domain/entities/agent_profile.py
- 로컬 에이전트의 capability, 버전, 플랫폼 정보를 표현합니다.
- FL 확장에서 중요합니다.

src/core/domain/value_objects/
- 작지만 의미가 명확한 불변 값 타입입니다.
- 예: Category, CohortKey, RoundId, ModelVersion

src/core/domain/value_objects/category.py
- Anxiety, Depression, Suicidal 같은 카테고리 표현입니다.

src/core/domain/value_objects/cohort_key.py
- 연령대 등 집계 단위를 식별하는 키입니다.

src/core/domain/value_objects/schema_version.py
- 계약 버전 관리를 위한 값 객체입니다.

src/core/domain/value_objects/round_id.py
- FL round 식별자입니다.

src/core/domain/value_objects/model_version.py
- 모델 버전 식별자입니다.

src/core/domain/policies/
- 판단 규칙, cohort 규칙, 학습 규칙 같은 정책 객체입니다.
- 모델과 분리된 규칙 계층입니다.

src/core/domain/policies/decision_policy.py
- Self-baseline + Peer norm + Persistence를 어떻게 결합할지 정의합니다.

src/core/domain/policies/cohort_policy.py
- cohort 생성 기준, 최소 지원 수, 집계 허용 조건을 정의합니다.

src/core/domain/policies/training_policy.py
- 어떤 에이전트가 어떤 학습 작업에 참여 가능한지, clipping/noise 정책이 무엇인지 등을 정의합니다.

src/core/ports/
- 외부 기술에 대한 인터페이스 모음입니다.
- 구현체를 교체 가능하게 만드는 경계입니다.

src/core/ports/embedding_port.py
- 임베딩 모델 호출 인터페이스입니다.

src/core/ports/translation_port.py
- 번역 모델 호출 인터페이스입니다.

src/core/ports/summary_repository_port.py
- WindowSummary 저장/조회 인터페이스입니다.

src/core/ports/norm_repository_port.py
- NormPack 저장/조회 인터페이스입니다.

src/core/ports/sync_client_port.py
- 중앙과 통신하는 동기화 클라이언트 인터페이스입니다.

src/core/ports/training_client_port.py
- FL 학습 작업을 수행하거나 업데이트를 보내는 클라이언트 인터페이스입니다.

src/core/ports/model_registry_port.py
- 모델 메타데이터와 버전 조회 인터페이스입니다.

src/services/
- 실제 유스케이스를 수행하는 서비스 계층입니다.
- 도메인 객체와 포트를 조합해서 흐름을 만듭니다.

src/services/local/
- 로컬 에이전트의 실제 처리 로직입니다.

src/services/local/preprocess_service.py
- 정규화, 언어 처리, 입력 정제 담당입니다.

src/services/local/embedding_service.py
- 텍스트를 임베딩으로 변환합니다.

src/services/local/scoring_service.py
- 프로토타입 유사도로 카테고리 점수를 계산합니다.

src/services/local/windowing_service.py
- 이벤트들을 시간 윈도우로 묶어 WindowSummary를 만듭니다.

src/services/local/baseline_service.py
- 개인 기준선을 계산하고 갱신합니다.

src/services/local/decision_service.py
- 로컬 최종 판단을 내립니다.
- 가장 핵심적인 로컬 정책 조합 지점입니다.

src/services/local/sync_service.py
- 중앙 업로드/다운로드를 관리합니다.
- summary 전송, norm 갱신, 재시도 흐름이 들어갑니다.

src/services/local/local_training_service.py
- FL 단계에서 로컬 학습 루프를 돌립니다.

src/services/local/privacy_guard_service.py
- 원문 비전송, payload 필터링, clipping/noise 같은 privacy guard 역할입니다.

src/services/central/
- 중앙 서버의 실제 처리 로직입니다.

src/services/central/ingestion_service.py
- 로컬에서 온 summary나 업데이트를 받아 검증하고 적재합니다.

src/services/central/aggregation_service.py
- cohort 단위 기본 집계를 수행합니다.

src/services/central/norm_pack_service.py
- 집계 결과를 NormPack으로 변환하고 배포 가능한 형태로 만듭니다.

src/services/central/agent_registry_service.py
- 에이전트 등록, heartbeat, capability 상태를 관리합니다.

src/services/central/round_manager_service.py
- FL round 생성, 참여 가능 상태, 종료 조건을 관리합니다.

src/services/central/update_aggregation_service.py
- 파라미터 업데이트를 집계합니다.
- Phase 5 이후 핵심 서비스입니다.

src/services/central/model_publication_service.py
- 새 모델 버전을 발행하고 manifest를 관리합니다.

src/services/central/privacy_orchestration_service.py
- secure aggregation, DP 정책, privacy enforcement 흐름을 중앙에서 조율합니다.

src/infrastructure/
- 실제 외부 기술 구현체입니다.
- 여기만 바꾸면 DB, HTTP 클라이언트, 모델 런타임을 교체할 수 있어야 합니다.

src/infrastructure/persistence/
- 물리 저장소 관련 구현입니다.
- SQLite, Postgres, 파일 저장 등 persistence 세부 구현이 들어갑니다.

src/infrastructure/repositories/
- core/ports를 실제 DB나 파일 시스템으로 구현한 repository 클래스들입니다.

src/infrastructure/transport/
- HTTP/gRPC/메시지 전송 같은 네트워크 어댑터입니다.

src/infrastructure/model_adapters/
- 실제 임베딩/번역/추론 엔진 연결 구현입니다.
- 모델 교체가 가장 자주 일어날 수 있는 곳입니다.

src/infrastructure/privacy/
- secure aggregation client/server adapter, DP noise mechanism, encryption helper 같은 privacy 기술 구현체입니다.

scripts/
- 개발/실험/시뮬레이션용 실행 스크립트 모음입니다.
- 재현 가능한 실험과 데모를 위해 둡니다.

scripts/seed_prototypes.py
- 카테고리 프로토타입 문장 세트 초기화 스크립트입니다.

scripts/run_local_demo.py
- 로컬 파이프라인만 빠르게 검증하는 데모 실행용입니다.

scripts/run_federated_simulation.py
- 여러 로컬 에이전트를 가정하고 중앙 집계/FL 흐름을 시뮬레이션하는 스크립트입니다.

tests/
- 검증 코드 전체입니다.

tests/unit/
- 순수 로직 단위 테스트입니다.
- 서비스나 도메인 정책을 빠르게 검증합니다.

tests/unit/core/
- 엔티티, value object, policy 같은 핵심 도메인 테스트입니다.

tests/unit/services/
- local/central 서비스 로직 테스트입니다.

tests/integration/
- 실제 연결 흐름 테스트입니다.

tests/integration/local/
- ingest -> preprocess -> score -> summary -> assessment 같은 로컬 흐름 검증입니다.

tests/integration/central/
- sync -> aggregation -> norm pack 발행 같은 중앙 흐름 검증입니다.

tests/integration/federation/
- 로컬 여러 개와 중앙이 같이 도는 end-to-end 흐름 검증입니다.
- 나중에 parameter FL round도 여기서 검증하게 됩니다.

정리하면 역할은 이렇게 보면 됩니다.
- core는 무엇을 다루는가
- services는 어떻게 처리하는가
- api는 어떻게 들어오는가
- infrastructure는 무엇으로 구현하는가

원하면 다음으로 바로 각 폴더를 실제 생성했을 때 어떤 파일부터 먼저 만들지까지 우선순위 순서로 이어서 정리해드리겠습니다.`

## User

`지금 내가 매번 명령하지 않아도 상시 skills는 작성 없는거지?
거기에 한국어로 답변 해야한다는것과 대화 전문을 notes에 작성해야 한다는것을 쓰자
그리고 현재도 너랑 대화하는거 전문이 저장 안되는 것 같아 저장되는게 없네`

## Assistant

`지금 상태는 스킬이 전역에 설치되어 있어도 항상 자동 적용되는 건 아닙니다. 명시적으로 $tracemind-project를 쓰거나, 요청이 설명과 강하게 맞을 때 트리거됩니다. 그래서 지금까지 notes에 자동 저장이 안 된 것도 자연스러운 상태였고, 그 동작을 바꾸려면 스킬 규칙 자체를 강화해야 합니다.`

## User

`이 모노레포 구조는 헷갈리지 않으려나 나중에 분리할떄.
main-server/src/
agent/src

이런 구조가 더 분리하기 쉬운거 아냐?
실험도 좋고?
모노레포는 분리해야 하는 부분이 겹칠 수 있는거 아닌가?`

## Assistant

`맞습니다. 지금 단계에서는 한 레포 안에 main-server / agent / shared를 나누는 방식이 더 명확하고, 나중에 별도 레포로 분리하기도 쉽다고 정리했습니다. 한 앱 트리 안에 local/central을 넣는 방식보다 실행 단위와 책임 경계가 더 선명해지고, shared에는 공통 계약과 도메인 규칙만 두는 쪽이 중복과 드리프트를 줄이기 좋다고 설명했습니다.`

## User

`최종 폴더 구조v2를 다시 재작성하고 plan 파일들도 수정 진행해`

## Assistant

`최종 폴더 구조 v2 기준으로 plan.md와 docs/project_execution_plan.md를 모두 main-server / agent / shared 구조로 수정했습니다. 그리고 바로 다음 액션 경로도 shared/src/domain/entities/window_summary.py, agent/src/services/windowing_service.py 같은 새 경로 기준으로 바꿨습니다.`

## User

`1. 예를들어 services 폴더라고 하면 세부 폴더 없이 저렇게 많은 파일들이 있어도 문제 없나?
2. 각 폴더와 파일 설명`

## Assistant

`지금 단계에서는 services 아래에 파일이 조금 많은 것 자체는 문제 없고, 역할 경계가 명확하면 그대로 가도 된다고 설명했습니다. 다만 서비스가 더 늘어나면 analytics / decision / federation / privacy 같은 하위 폴더로 2차 분리하는 게 좋다고 했고, 이어서 main-server / agent / shared v2 구조 기준으로 각 폴더와 파일이 담당할 책임을 상세히 정리했습니다.`

## User

`크롬 확장 프로그램을 통해 입력 텍스트를 가로채서 가져와서 번역 모델에 넣고 임베딩 모델(데이터 셋을 가지고 미리 4개의 카테고리로 분류)에 넣어서 할건데 그건 어디에 들어가야 하는거지`

## Assistant

`그 역할은 agent 쪽이라고 정리했습니다. 정확히는 크롬 확장 프로그램은 브라우저 입력 수집 어댑터이고, 실제 번역/임베딩/분류/요약 파이프라인은 agent 내부 서비스에 둬야 한다고 설명했습니다. 그래서 agent/chrome-extension은 content/background/popup/bridge 같은 입력 수집 계층만 가지고, 핵심 분석 로직은 agent/src/services 와 agent/src/infrastructure/model_adapters 아래에 두는 방향을 제안했습니다.`

## User

`수정 진행해
그리고 개발은 확장 프로그램은 이따가고 임베딩 모델 선정한것과 데이터셋을 가져와서 먼저 넣어놓고 진행하는게 좋나? 아니면 그건 나중에 따로 하고 계약부터 진행하는게 맞나?`

## Assistant

`plan.md와 docs/project_execution_plan.md를 v2.1 구조로 수정했습니다. agent/chrome-extension을 브라우저 입력 수집 어댑터로 추가했고, 모델/데이터셋은 병렬로 준비하되 계약보다 먼저 파이프라인에 박아 넣지 말고 WindowSummary, NormPack, sync contract 같은 계약을 먼저 고정한 뒤 연결하는 순서가 맞다고 정리했습니다.`

## User

`다음번에 바로 이어서 할 수 있는 프롬포트 작성해`
