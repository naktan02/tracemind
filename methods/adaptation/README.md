# Adaptation Methods

`methods/adaptation/`은 학습 adapter 적용 core를 둔다.

local update backend의 concrete 구현과 registry는 `methods/adaptation/`이 소유한다.
새 adaptation family나 backend를 추가하기 위해
`agent/src/services/training/backends/training/` compatibility path를 다시 만들지
않는다.

## 하위 패키지 지도

- `diagonal_scale/`: diagonal-scale heuristic local update 계산 core와 family별
  aggregation adapter
- `classification/`: modality-independent classification primitive와 feature-head
  aggregation/scoring projection
- `local_update_backend.py`: agent가 호출하는 local update backend port
- `local_update_registry.py`: method-owned local update backend lookup/catalog facade
- `server_update_materialization.py`: adapter family별 서버 materialization
  preflight를 찾아 실행하는 dispatcher
- `privacy_guards/`: shared adapter update clipping/DP policy core와 registry
- `peft_adapters/`: LoRA/DoRA 같은 PEFT mechanism builder와 registry.
  classifier/task payload 의미는 소유하지 않는다.
- `lora_classifier/`: frozen backbone + LoRA/PEFT adapter + classifier head
  재사용 scaffold
- `text_classifier/`: text classifier task family의 장기 목표 adaptation 구조.
  `peft_encoder/`, aggregation projection 경계를 분리한다.
- `query_classifier_adaptation/`: query-domain LoRA/classifier 중앙 실험의
  token-batch 입력 glue와 weak/strong view row 해석 helper

rank, alpha, target module 같은 실행 파라미터는 code folder가 아니라 config에서
선택한다.

`methods/adaptation/<family>/aggregation/fedavg.py` 같은 family별 aggregation
module은 payload delta 해석과 next-state materialization을 맡는 method-owned seam이다.
generic FedAvg 산술과 strategy wiring은 `methods/federated/aggregation/fedavg/`에
두고, family 상세는 `main_server`가 아니라 해당 adapter family package에 둔다.

서버 update preflight도 같은 기준을 따른다. `methods/adaptation/server_update_*`
dispatcher는 `adapter_kind` 기반 import/lookup만 맡고, state compatibility,
artifact ref 해석, inline delta 요구사항은
`methods/adaptation/<family>/server_preflight.py`가 소유한다.

privacy guard도 같은 기준을 따른다. `agent`는 selected guard를 실행 흐름에 연결하고,
guard 이름, adapter-kind 지원 범위, clipping 계산은 `methods/adaptation/privacy_guards/`
가 소유한다.

## 추가 기준

DoRA, IA3, linear probe, full fine-tune은 실제 실험/런타임 경계가 필요해질 때
별도 method로 추가한다. 이름만 미리 열어 두지 않는다.
