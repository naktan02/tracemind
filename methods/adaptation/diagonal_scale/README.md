# Diagonal Scale Adaptation

`methods/adaptation/diagonal_scale/`는 diagonal-scale adapter family의 재사용 가능한
학습/update 계산 core를 둔다.

## 책임

- accepted pseudo-label example의 confidence 가중 평균 방향 계산
- `VectorAdapterDelta` 생성
- `fedavg.py`에서 diagonal-scale delta를 FedAvg 공통 산술로 평균
- `fedavg_projection.py`에서 diagonal-scale state/update payload를 family FedAvg core
  입력으로 변환
- diagonal-scale update metric 추출
- backend-specific objective extras를 method config로 정규화
- `diagonal_scale_heuristic` local update backend registration

## 제외

- agent-local query buffer와 raw text 접근
- local artifact 저장
- privacy guard, secure update codec, update upload

위 runtime glue는 `agent/src/services/training/`과 `agent/src/services/federation/`에
남긴다.
