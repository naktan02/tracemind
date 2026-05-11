"""중앙 Query SSL weak/strong text view materialization CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.datasets.lib.query_ssl_views import (
    NllbBacktranslationRuntimeConfig,
    QuerySslViewArtifacts,
    build_nllb_backtranslation_candidate_pair_builder,
    materialize_query_ssl_backtranslation_views,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Query SSL labeled/unlabeled text views with two NLLB "
            "backtranslation strong candidates."
        )
    )
    parser.add_argument(
        "--split-dir",
        required=True,
        type=Path,
        help="Directory containing labeled_train.jsonl and unlabeled_pool.jsonl.",
    )
    parser.add_argument(
        "--split-name",
        required=True,
        help="Stable split name used in output paths and manifest.",
    )
    parser.add_argument(
        "--augmenter-name",
        default="backtranslation_nllb_en_de_fr_usb_v1",
        help="Stable augmenter name used in output paths and manifest.",
    )
    parser.add_argument("--source-lang", default="eng_Latn")
    parser.add_argument("--pivot-languages", nargs=2, default=["deu_Latn", "fra_Latn"])
    parser.add_argument(
        "--model-id",
        default="facebook/nllb-200-distilled-600M",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of source rows translated and appended per checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache/query_ssl_views/model_cache",
        help="Model cache directory for the NLLB runtime.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing .tmp JSONL progress. Use with --overwrite to restart.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output/progress files before materialization.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/artifacts/query_ssl_views"),
        help=(
            "Root directory where view artifacts are written. Prefer a "
            "dataset-scoped data/datasets/<dataset_id>/views path in Hydra presets."
        ),
    )
    return parser.parse_args()


def run_query_ssl_view_materialization(
    *,
    split_dir: Path,
    split_name: str,
    augmenter_name: str,
    source_lang: str,
    pivot_languages: tuple[str, str],
    model_id: str,
    revision: str,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    torch_dtype: str,
    cache_dir: str | None,
    local_files_only: bool,
    chunk_size: int,
    output_root: Path,
    resume: bool,
    overwrite: bool,
) -> QuerySslViewArtifacts:
    """CLI와 Hydra wrapper가 공유하는 Query SSL view materialization 실행부."""

    runtime_config = NllbBacktranslationRuntimeConfig(
        source_lang=source_lang,
        pivot_languages=pivot_languages,
        model_id=model_id,
        revision=revision,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return materialize_query_ssl_backtranslation_views(
        split_dir=split_dir,
        split_name=split_name,
        augmenter_name=augmenter_name,
        output_root=output_root,
        augmenter_manifest=runtime_config.to_manifest(),
        candidate_pair_builder=build_nllb_backtranslation_candidate_pair_builder(
            runtime_config
        ),
        chunk_size=chunk_size,
        resume=resume,
        overwrite=overwrite,
    )


def main() -> None:
    args = parse_args()
    artifacts = run_query_ssl_view_materialization(
        split_dir=args.split_dir,
        split_name=str(args.split_name),
        augmenter_name=str(args.augmenter_name),
        source_lang=str(args.source_lang),
        pivot_languages=(str(args.pivot_languages[0]), str(args.pivot_languages[1])),
        model_id=str(args.model_id),
        revision=str(args.revision),
        device=str(args.device),
        batch_size=int(args.batch_size),
        max_new_tokens=int(args.max_new_tokens),
        torch_dtype=str(args.torch_dtype),
        cache_dir=None if args.cache_dir == "" else str(args.cache_dir),
        local_files_only=bool(args.local_files_only),
        chunk_size=int(args.chunk_size),
        output_root=args.output_root,
        resume=not bool(args.no_resume),
        overwrite=bool(args.overwrite),
    )

    print(f"labeled_train_with_views_jsonl={artifacts.labeled_train_with_views_jsonl}")
    print(
        f"unlabeled_pool_with_views_jsonl={artifacts.unlabeled_pool_with_views_jsonl}"
    )
    print(f"manifest_json={artifacts.manifest_json}")
    print(f"summary_json={artifacts.summary_json}")
    print(f"progress_json={artifacts.progress_json}")


if __name__ == "__main__":
    main()
