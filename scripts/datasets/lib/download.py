"""dataset 다운로드 재사용 함수."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile


def build_default_output_path(
    *,
    dataset_id: str,
    split: str,
    output_dir: Path,
) -> Path:
    """dataset_id와 split을 포함한 기본 출력 경로를 만든다."""
    safe_name = dataset_id.replace("/", "_")
    return output_dir / f"{safe_name}__{split}.csv"


def download_huggingface_dataset_to_csv(
    *,
    dataset_id: str,
    split: str,
    output_dir: Path,
    cache_dir: Path,
    data_file: str | None = None,
    output_path: Path | None = None,
    revision: str | None = None,
) -> Path:
    """HuggingFace 데이터셋 split 하나를 CSV로 저장한다."""
    from datasets import load_dataset

    resolved_output_path = output_path or build_default_output_path(
        dataset_id=dataset_id,
        split=split,
        output_dir=output_dir,
    )
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    load_kwargs: dict[str, object] = {
        "path": dataset_id,
        "split": split,
        "cache_dir": str(cache_dir),
    }
    if data_file is not None:
        load_kwargs["data_files"] = {split: data_file}
    if revision is not None:
        load_kwargs["revision"] = revision

    print(
        f"다운로드 시작: dataset={dataset_id} split={split} "
        f"data_file={data_file or '-'}",
        flush=True,
    )
    ds = load_dataset(**load_kwargs)
    ds.to_csv(str(resolved_output_path), index=False)
    print(f"저장 완료: {resolved_output_path} ({len(ds)} rows)", flush=True)
    return resolved_output_path


def download_kaggle_dataset_file_to_csv(
    *,
    dataset_ref: str,
    data_file: str,
    output_path: Path,
    dataset_version_number: int | None = None,
    download_url: str | None = None,
    username: str | None = None,
    key: str | None = None,
) -> Path:
    """Kaggle dataset ZIP에서 지정 CSV 파일 하나를 raw CSV로 저장한다."""

    normalized_ref = dataset_ref.strip()
    normalized_data_file = data_file.strip()
    if not normalized_ref:
        raise ValueError("dataset_ref must not be empty.")
    if not normalized_data_file:
        raise ValueError("data_file must not be empty.")

    resolved_output_path = output_path
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)

    auth_username = username or os.environ.get("KAGGLE_USERNAME")
    auth_key = key or os.environ.get("KAGGLE_KEY")
    candidate_urls = _build_kaggle_download_urls(
        dataset_ref=normalized_ref,
        dataset_version_number=dataset_version_number,
        download_url=download_url,
    )

    last_error: Exception | None = None
    for url in candidate_urls:
        try:
            payload = _download_url_bytes(
                url,
                username=auth_username,
                key=auth_key,
            )
            _extract_zip_member_to_file(
                zip_payload=payload,
                data_file=normalized_data_file,
                output_path=resolved_output_path,
            )
            print(
                "저장 완료: "
                f"{resolved_output_path} (kaggle={normalized_ref}, file={data_file})",
                flush=True,
            )
            return resolved_output_path
        except (HTTPError, BadZipFile, FileNotFoundError, ValueError) as exc:
            last_error = exc

    credential_hint = (
        "Kaggle download failed. Set KAGGLE_USERNAME/KAGGLE_KEY or download the "
        "dataset ZIP manually and place the CSV at the configured raw path."
    )
    if last_error is None:
        raise RuntimeError(credential_hint)
    raise RuntimeError(f"{credential_hint} Last error: {last_error}") from last_error


def _build_kaggle_download_urls(
    *,
    dataset_ref: str,
    dataset_version_number: int | None,
    download_url: str | None,
) -> tuple[str, ...]:
    urls: list[str] = []
    if download_url is not None and download_url.strip():
        urls.append(download_url.strip())

    api_url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_ref}"
    if dataset_version_number is not None:
        api_url = f"{api_url}?datasetVersionNumber={int(dataset_version_number)}"
    urls.append(api_url)
    return tuple(dict.fromkeys(urls))


def _download_url_bytes(
    url: str,
    *,
    username: str | None,
    key: str | None,
) -> bytes:
    request = Request(url, headers={"User-Agent": "tracemind-dataset-pipeline/1.0"})
    if username and key:
        token = base64.b64encode(f"{username}:{key}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")
    with urlopen(request) as response:  # noqa: S310 - configured dataset URL.
        return response.read()


def _extract_zip_member_to_file(
    *,
    zip_payload: bytes,
    data_file: str,
    output_path: Path,
) -> None:
    try:
        archive = ZipFile(BytesIO(zip_payload))
    except BadZipFile as exc:
        preview = zip_payload[:120].decode("utf-8", errors="replace")
        raise BadZipFile(
            "Kaggle download did not return a ZIP payload. "
            f"payload_preview={preview!r}"
        ) from exc

    with archive:
        member_name = _resolve_zip_member_name(archive=archive, data_file=data_file)
        with archive.open(member_name) as source_file, output_path.open(
            "wb"
        ) as output_file:
            output_file.write(source_file.read())


def _resolve_zip_member_name(*, archive: ZipFile, data_file: str) -> str:
    normalized_data_file = data_file.strip()
    for member_name in archive.namelist():
        if member_name == normalized_data_file:
            return member_name
    for member_name in archive.namelist():
        if Path(member_name).name == normalized_data_file:
            return member_name
    raise FileNotFoundError(
        f"Kaggle ZIP does not contain {data_file!r}. "
        f"members={archive.namelist()[:10]}"
    )
