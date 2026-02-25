from pathlib import Path
from urllib.parse import urlparse


def extract_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    segments = _get_non_empty_path_segments(parsed.path)
    raw_name = segments[-1] if segments else parsed.hostname or "output"
    return _strip_file_extension(raw_name)


def ensure_output_directories(output_base: Path, name: str) -> Path:
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / name).mkdir(parents=True, exist_ok=True)
    return output_base


def write_markdown_file(output_base: Path, name: str, markdown_content: str) -> Path:
    output_path = output_base / f"{name}.md"
    output_path.write_text(markdown_content, encoding="utf-8")
    return output_path


def get_default_output_dir() -> Path:
    return Path("./outputs")


def _get_non_empty_path_segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment]


def _strip_file_extension(name: str) -> str:
    stem = Path(name).stem
    return stem if stem else name
