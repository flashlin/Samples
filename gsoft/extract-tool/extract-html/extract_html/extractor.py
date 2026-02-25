from dataclasses import dataclass, field
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag

_REQUEST_TIMEOUT = 15
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_NON_CONTENT_TAGS = ["script", "style", "nav", "header", "footer", "aside", "iframe"]
_CONTENT_SELECTORS = ["article", "main", ".markdown-body"]


@dataclass
class ExtractionResult:
    html_content: str
    image_urls: list[str] = field(default_factory=list)
    base_url: str = ""


def extract_from_url(url: str) -> ExtractionResult:
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "lxml")
    _remove_non_content_elements(soup)
    content = _find_main_content(soup)
    image_urls = _collect_image_urls(content, url)
    return ExtractionResult(
        html_content=str(content),
        image_urls=image_urls,
        base_url=url,
    )


def _fetch_html(url: str) -> str:
    response = requests.get(url, timeout=_REQUEST_TIMEOUT, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    return response.text


def _remove_non_content_elements(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_NON_CONTENT_TAGS):
        tag.decompose()


def _find_main_content(soup: BeautifulSoup) -> Tag:
    for selector in _CONTENT_SELECTORS:
        element = soup.select_one(selector)
        if element:
            return element
    return soup.find("body") or soup


def _collect_image_urls(content_tag: Tag, base_url: str) -> list[str]:
    return [
        _resolve_image_url(img["src"], base_url)
        for img in content_tag.find_all("img")
        if _is_valid_image_src(img.get("src", ""))
    ]


def _is_valid_image_src(src: str) -> bool:
    return bool(src) and not src.startswith("data:")


def _resolve_image_url(src: str, base_url: str) -> str:
    return urljoin(base_url, src)
