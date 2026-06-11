"""CoMatch USB 원본 출처와 v1 이식 범위."""

from __future__ import annotations

COMATCH_ORIGINAL_REPOSITORY = "https://github.com/microsoft/Semi-supervised-learning"
COMATCH_ORIGINAL_COMMIT = "1ef4cbebcc0b368158315aeb425053858cf6c845"
COMATCH_ORIGINAL_PATH = "semilearn/algorithms/comatch/comatch.py"


def comatch_original_parameter_mapping() -> dict[str, str]:
    """TraceMind CoMatch v1 parameter와 USB 원본 인자의 대응을 반환한다."""

    return {
        "temperature": "T",
        "p_cutoff": "p_cutoff",
        "contrast_p_cutoff": "contrast_p_cutoff",
        "queue_batch": "queue_batch",
        "smoothing_alpha": "smoothing_alpha",
        "da_len": "da_len",
        "proj_size": "proj_size",
        "lambda_c": "contrast_loss_ratio",
        "lambda_u": "lambda_u",
    }


COMATCH_V1_INTENTIONAL_DEVIATIONS = (
    "USB AlgorithmBase, distributed concat_all_gather, EMA model, EPASS "
    "multi-projection은 v1 범위에서 제외한다.",
    "TraceMind는 text Query SSL batch metadata로 queue_batch * "
    "(labeled_batch_size + unlabeled_batch_size) memory row capacity를 계산한다.",
    "TraceMind는 USB의 초기 iteration warm-up을 global optimizer step 기준 "
    "global_step > queue_batch로 옮긴다.",
    "Projection head는 public classifier/update family가 아니라 algorithm-local "
    "auxiliary trainable module로 checkpoint한다.",
)
