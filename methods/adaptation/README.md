# Adaptation Methods

`methods/adaptation/`은 학습 adapter 적용 core를 둔다.

local update backend의 concrete 구현은 `methods/adaptation/<family>/training_backend.py`
에 둔다. `agent/src/services/training/backends/training/`은 compatibility facade만
남기며, 새 adaptation family나 backend를 추가하기 위해 agent training 폴더에
method-specific 파일을 만들지 않는다.

## 하위 패키지 지도

- `diagonal_scale/`: diagonal-scale heuristic local update 계산 core
- `classifier_head/`: classifier-head shared adapter family projection
- `local_update_backend.py`: agent가 호출하는 local update backend port
- `local_update_registry.py`: method-owned local update backend lookup/catalog facade
- `peft/`: PEFT adapter builder protocol과 registry
- `lora/`: LoRA/RSLoRA builder core
- `lora_classifier/`: frozen backbone + LoRA/PEFT adapter + classifier head
  재사용 scaffold
- `query_classifier_adaptation/`: query-domain LoRA/classifier 중앙 실험의
  token-batch 입력 glue와 기존 import compatibility shim

rank, alpha, target module 같은 실행 파라미터는 code folder가 아니라 config에서
선택한다.

`fedavg_projection.py` 같은 aggregation projection은 family별 payload를
`methods/federated/aggregation/*` core 입력으로 바꾸는 method-owned seam이다. 이
projection은 `main_server`가 아니라 해당 adapter family package에 둔다.

## 추가 기준

DoRA, IA3, linear probe, full fine-tune은 실제 실험/런타임 경계가 필요해질 때
별도 method로 추가한다. 이름만 미리 열어 두지 않는다.
