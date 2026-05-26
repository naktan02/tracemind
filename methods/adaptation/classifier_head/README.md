# Classifier Head Adaptation

`methods/adaptation/classifier_head/`는 기존 direct import 호환성만 유지하는
legacy 경로다. 새 source of truth는 `methods/adaptation/text_classifier/feature_head/`
와 `methods/adaptation/text_classifier/aggregation/`이다.

## 책임

- 기존 `bootstrap.py`, `scoring.py`, `aggregation/fedavg.py` direct path를 named-symbol
  compatibility shim으로 유지
- business rule, source-of-truth 상수, registry metadata를 새 경로에 둠

## 제외

- 새 feature-head 학습/평가/scoring primitive 소유
- FedAvg 산술과 projection 구현 소유
- shared payload shape 정의
