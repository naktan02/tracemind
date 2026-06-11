"""USB MixMatch provenance와 TraceMind 이식 범위."""

USB_MIXMATCH_SOURCE = {
    "repo": "microsoft/Semi-supervised-learning",
    "commit": "1ef4cbebcc0b368158315aeb425053858cf6c845",
    "path": "semilearn/algorithms/mixmatch/mixmatch.py",
}

PRESERVED_CORE = (
    "unlabeled weak/strong 두 view 확률 평균으로 pseudo-label을 만든다.",
    "temperature T로 pseudo-label을 sharpening한다.",
    "labeled one-hot label과 unlabeled sharpened soft label 두 벌을 concat한다.",
    "USB mixup_one_target(..., is_bias=True)와 같은 Beta MixUp을 적용한다.",
    "mixed labeled chunk에는 soft-target CE를, mixed unlabeled chunk에는 "
    "softmax 확률 MSE를 적용한다.",
    "unsup_warm_up * num_train_iter 기준 linear ramp-up으로 lambda_u를 키운다.",
)

TRACEMIND_ADAPTATIONS = (
    "USB NLP config는 mixup_manifold=True를 사용한다. TraceMind도 텍스트 token id를 "
    "섞지 않고 PEFT text encoder의 classifier 직전 feature를 섞는다.",
    "USB BERT의 only_fc=True 경로는 TraceMind의 model.classifier 직접 호출로 대응한다.",
    "USB uratio=1 전제를 보존하기 위해 labeled/unlabeled batch size가 다르면 "
    "명시적으로 실패한다.",
    "USB BN freeze와 AMP AlgorithmBase glue는 TraceMind text PEFT runtime의 "
    "공통 optimizer/lifecycle에 위임한다.",
)
