from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import requests

ProgressCallback = Callable[[str, int, int], None]

_DOWNLOAD_TIMEOUT = 15


@dataclass
class DownloadResult:
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def download_images(
    image_mappings: dict[str, str],
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
) -> DownloadResult:
    result = DownloadResult()
    total = len(image_mappings)

    for index, (url, relative_path) in enumerate(image_mappings.items(), start=1):
        save_path = output_dir / relative_path
        _ensure_parent_directory(save_path)

        filename = save_path.name
        if _download_single_image(url, save_path):
            result.succeeded.append(filename)
        else:
            result.failed.append(url)

        if progress_callback:
            progress_callback(filename, index, total)

    return result


def _ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _download_single_image(url: str, save_path: Path) -> bool:
    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT, stream=True)
        response.raise_for_status()
        _write_response_to_file(response, save_path)
        return True
    except requests.RequestException:
        return False


def _write_response_to_file(response: requests.Response, save_path: Path) -> None:
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
