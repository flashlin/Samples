from dataclasses import dataclass

import numpy as np

from .engines import SpeechRecognizer, Translator
from .text_normalizer import TextLanguage, classify_language, terms_present, to_traditional


@dataclass(frozen=True)
class CaptionText:
    text: str
    language: TextLanguage
    translation: str | None
    display: str


class CaptionPipeline:
    def __init__(self, recognizer: SpeechRecognizer, translator: Translator, keep_terms: list[str], bilingual_english: bool) -> None:
        self._recognizer = recognizer
        self._translator = translator
        self._keep_terms = keep_terms
        self._bilingual_english = bilingual_english

    @property
    def engine_names(self) -> dict[str, str]:
        return {"asr": self._recognizer.name, "translate": self._translator.name}

    def transcribe(self, samples: np.ndarray, translate: bool = True) -> CaptionText:
        return self.caption_text(self._recognizer.transcribe(samples), translate)

    def caption_text(self, raw_text: str, translate: bool = True) -> CaptionText:
        text = to_traditional(raw_text.strip())
        language = classify_language(text)
        if language is not TextLanguage.ENGLISH or not translate:
            return CaptionText(text=text, language=language, translation=None, display=text)
        translation = self.translate_english(text)
        display = f"{text}\n{translation}" if self._bilingual_english else translation
        return CaptionText(text=text, language=language, translation=translation, display=display)

    def translate_english(self, text: str) -> str:
        keep = terms_present(text, self._keep_terms)
        return to_traditional(self._translator.translate(text, keep))
