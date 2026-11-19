import re
from enum import StrEnum

from opencc import OpenCC

_TO_TRADITIONAL = OpenCC("s2tw")
_HAN = re.compile(r"[㐀-鿿豈-﫿]")
_LATIN_WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
MIN_ENGLISH_WORDS = 2


class TextLanguage(StrEnum):
    ENGLISH = "en"
    CHINESE = "zh"
    UNKNOWN = "unknown"


def to_traditional(text: str) -> str:
    return _TO_TRADITIONAL.convert(text)


def classify_language(text: str) -> TextLanguage:
    if _HAN.search(text):
        return TextLanguage.CHINESE
    if len(_LATIN_WORD.findall(text)) >= MIN_ENGLISH_WORDS:
        return TextLanguage.ENGLISH
    return TextLanguage.UNKNOWN


def terms_present(text: str, terms: list[str]) -> list[str]:
    found: list[str] = []
    for term in terms:
        pattern = rf"(?<![A-Za-z]){re.escape(term)}(?:e?s)?(?![A-Za-z])"
        for surface in re.findall(pattern, text, flags=re.IGNORECASE):
            if surface not in found:
                found.append(surface)
    return found
