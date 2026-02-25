import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

import markdownify

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"}


@dataclass
class ConversionResult:
    markdown: str
    image_mappings: dict[str, str] = field(default_factory=dict)


def convert_html_to_markdown(
    html_content: str, image_urls: list[str], output_name: str
) -> ConversionResult:
    image_mappings = _build_image_mappings(image_urls, output_name)
    modified_html = _replace_image_sources(html_content, image_mappings)
    raw_markdown = markdownify.markdownify(
        modified_html,
        heading_style="ATX",
        strip=["script", "style"],
    )
    cleaned = _clean_markdown(raw_markdown)
    return ConversionResult(markdown=cleaned, image_mappings=image_mappings)


def _build_image_mappings(image_urls: list[str], output_name: str) -> dict[str, str]:
    mappings: dict[str, str] = {}
    seen_names: dict[str, int] = {}

    for index, url in enumerate(image_urls, start=1):
        filename = _extract_filename_from_url(url, index)
        filename = _resolve_duplicate_filename(filename, seen_names)
        mappings[url] = f"{output_name}/{filename}"

    return mappings


def _resolve_duplicate_filename(filename: str, seen_names: dict[str, int]) -> str:
    if filename not in seen_names:
        seen_names[filename] = 0
        return filename

    seen_names[filename] += 1
    stem, _, suffix = filename.rpartition(".")
    new_name = f"{stem}_{seen_names[filename]}.{suffix}" if suffix else f"{filename}_{seen_names[filename]}"
    seen_names[new_name] = 0
    return new_name


def _extract_filename_from_url(url: str, index: int) -> str:
    parsed = urlparse(url)
    path_segments = parsed.path.rstrip("/").split("/")
    last_segment = path_segments[-1] if path_segments else ""
    if _looks_like_image_filename(last_segment):
        return last_segment
    return f"image_{index:03d}.png"


def _looks_like_image_filename(name: str) -> bool:
    if not name:
        return False
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return suffix in IMAGE_EXTENSIONS


def _replace_image_sources(html: str, image_mappings: dict[str, str]) -> str:
    result = html
    for original_url, local_path in image_mappings.items():
        result = result.replace(f'src="{original_url}"', f'src="{local_path}"')
        result = result.replace(f"src='{original_url}'", f"src='{local_path}'")
    return result


def _clean_markdown(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
