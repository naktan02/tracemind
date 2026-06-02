"""USB ReMixMatch provenance와 TraceMind 이식 범위."""

USB_REMIXMATCH_SOURCE = {
    "repo": "microsoft/Semi-supervised-learning",
    "commit": "1ef4cbebcc0b368158315aeb425053858cf6c845",
    "path": "semilearn/algorithms/remixmatch/remixmatch.py",
}

PRESERVED_CORE = (
    "unlabeled weak view probability에 GT-target EMA distribution alignment를 "
    "적용한다.",
    "aligned probability를 temperature T로 sharpening한다.",
    "labeled one-hot label과 unlabeled sharpened label 세 벌을 concat한다.",
    "USB mixup_one_target(..., is_bias=True) 방식으로 manifold MixUp을 적용한다.",
    "mixed labeled chunk에는 soft CE, mixed unlabeled chunks에는 probability MSE를 "
    "적용한다.",
    "첫 strong view logits에는 sharpened weak target CE(u1_loss)를 추가한다.",
    "unsup_warm_up 기반 ramp-up으로 unsup_loss와 u1_loss weight를 키운다.",
)

TRACEMIND_ADAPTATIONS = (
    "USB NLP ReMixMatch config는 mixup_manifold=True, rot_loss_ratio=0.0을 사용한다. "
    "TraceMind도 텍스트 token id를 섞지 않고 PEFT text encoder의 classifier 직전 "
    "feature를 섞는다.",
    "USB BERT의 only_fc=True 경로는 TraceMind의 model.classifier 직접 호출로 대응한다.",
    "이미지 회전 auxiliary head/loss는 텍스트 runtime에 맞지 않으므로 "
    "rot_loss_ratio=0.0만 지원한다.",
    "USB uratio=1 전제를 보존하기 위해 labeled/unlabeled batch size가 다르면 "
    "명시적으로 실패한다.",
    "USB EMA model, AMP, distributed, BN freeze glue는 TraceMind 공통 trainer "
    "lifecycle에 위임한다.",
)
