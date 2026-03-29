# 2026-03-29 Dataset Pipeline Run

## User

지금 uv sync로 다운까지 완료했어 이제 ourafla 데이터셋 다운로드 하고 train하고 test둘다 가져와서 train을 split할거야 그리고 train으로 각 카테고리 임베딩 점수 맞출거야 지금 그걸 위한 코드 준비는  다 되어있을거야
그리고 cssrs는 일단 다운로드만 받고 매칭은 일단 시키지 않을거야
https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset?utm_source=chatgpt.com

https://www.kaggle.com/datasets/av9ash/labelled-reddit-suicidewatch-posts-cssr-s

매번 내가 이렇게 링크 주는게 아니라 해당 데이터셋 정보도 넣을 수 있는 yaml 같은거 해놓고 데이터셋 쉽게 추가 가능하도록 세팅 되어야 하지 않나?

일단 그렇게 해서 임베딩해서 카테고리 결과 임베딩 점수를 어디에 저장하지 일단 실행하자

## Assistant

- `data/datasets/registry.yaml`을 추가해 데이터셋 소스, 매핑 파일, split, prototype 빌드 단계를 선언형으로 관리하도록 구성했다.
- `scripts/run_dataset_pipeline.py`를 추가해 `download -> map -> split -> prototype` 파이프라인을 레지스트리 기반으로 실행하게 만들었다.
- `scripts/download_dataset.py`는 split별 출력 파일명이 겹치지 않도록 수정했고, Hugging Face repo 내부 파일명을 직접 지정할 수 있게 확장했다.
- `scripts/split_labeled_query_set.py`, `scripts/seed_prototypes.py`는 파이프라인 러너에서 재사용할 수 있도록 함수형 엔트리포인트를 추가했다.
- `ourafla`는 Hugging Face에서 `train`과 `test`를 모두 내려받고, train JSONL을 stratified split한 뒤 `mixedbread-ai/mxbai-embed-large-v1`로 카테고리 centroid prototype pack을 생성했다.
- `cssrs`는 현재 단계 요구에 맞춰 다운로드만 수행했고, 레지스트리에는 Kaggle 원본 링크와 Hugging Face 미러 링크를 함께 기록했다.
- 산출물은 `data/raw/`, `data/processed/labeled_query_sets/`, `data/processed/splits/`, `data/processed/prototype_packs/`, `data/processed/pipeline_runs/`에 저장되도록 정리했다.
- ourafla 실행 결과 manifest: `data/processed/pipeline_runs/ourafla__20260328T153949Z.manifest.json`
- cssrs 실행 결과 manifest: `data/processed/pipeline_runs/cssrs__20260328T153952Z.manifest.json`
- ourafla prototype pack: `data/processed/prototype_packs/ourafla_mxbai_20260328T153818Z.json`
- ourafla split manifest 기준 train rows는 44,651, validation rows는 4,961이고, prototype pack은 `anxiety`, `depression`, `normal`, `suicidal` 4개 카테고리와 1024차원 centroid를 가진다.
- 로컬 검증은 `compileall`과 registry smoke test로 수행했고, 현재 `.venv`에는 `pytest`와 `ruff`가 없어 해당 도구 기반 검증은 실행하지 못했다.
