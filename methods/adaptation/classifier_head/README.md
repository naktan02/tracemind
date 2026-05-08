# Classifier Head Adaptation

`methods/adaptation/classifier_head/`는 classifier-head shared adapter family의
method-owned FedAvg core/projection을 둔다.

## 책임

- classifier-head delta를 FedAvg 공통 산술로 평균
- `ClassifierHeadState`와 `ClassifierHeadDelta`를 family FedAvg core 입력으로 변환
- label schema와 embedding dimension consistency 검증
- FedAvg 결과를 다음 classifier-head shared state로 materialize
- classifier-head shared state 기반 logits scoring backend

## 제외

- shared payload shape 정의는 `shared/src/contracts/adapter_contract_families/`가 맡는다.
- round lifecycle, update storage, publication은 `main_server`가 맡는다.
- agent-local scoring service orchestration은 `agent`가 맡는다.
