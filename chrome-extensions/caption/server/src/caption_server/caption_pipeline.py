from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SAMPLE_RATE
from .engines import Punctuator, SpeechRecognizer, Translator
from .sentence_segmenter import merge_punctuation, split_sentences, time_sentences
from .text_normalizer import TextLanguage, classify_language, terms_present, to_traditional


@dataclass(frozen=True)
class CaptionText:
    text: str
    language: TextLanguage
    translation: str | None
    display: str


@dataclass(frozen=True)
class CaptionSegment:
    start: float
    end: float
    caption: CaptionText


@dataclass(frozen=True)
class ChunkCaption:
    caption: CaptionText
    segments: list[CaptionSegment]


@dataclass(frozen=True)
class PipelineOptions:
    keep_terms: list[str]
    bilingual_english: bool
    max_sentence_units: int


class CaptionPipeline:
    def __init__(self, recognizer: SpeechRecognizer, punctuator: Punctuator, translator: Translator, options: PipelineOptions) -> None:
        self._recognizer = recognizer
        self._punctuator = punctuator
        self._translator = translator
        self._options = options

    @property
    def engine_names(self) -> dict[str, str]:
        return {"asr": self._recognizer.name, "punctuation": self._punctuator.name, "translate": self._translator.name}

    def transcribe(self, samples: np.ndarray, final: bool = True) -> ChunkCaption:
        recognition = self._recognizer.recognize(samples, with_timestamps=final)
        text = to_traditional(recognition.text)
        punctuated = merge_punctuation(text, self._punctuator.punctuate(text))
        sentences = split_sentences(punctuated, self._options.max_sentence_units)
        timed = time_sentences(sentences, recognition.tokens, len(samples) / TARGET_SAMPLE_RATE)
        segments = [CaptionSegment(segment.start, segment.end, self.caption_text(segment.text, translate=final)) for segment in timed]
        whole = CaptionText(
            text=punctuated,
            language=classify_language(punctuated),
            translation=None,
            display="\n".join(segment.caption.display for segment in segments),
        )
        return ChunkCaption(caption=whole, segments=segments)

    def caption_text(self, raw_text: str, translate: bool = True) -> CaptionText:
        text = to_traditional(raw_text.strip())
        language = classify_language(text)
        if language is not TextLanguage.ENGLISH or not translate:
            return CaptionText(text=text, language=language, translation=None, display=text)
        translation = self.translate_english(text)
        display = f"{text}\n{translation}" if self._options.bilingual_english else translation
        return CaptionText(text=text, language=language, translation=translation, display=display)

    def translate_english(self, text: str) -> str:
        keep = terms_present(text, self._options.keep_terms)
        return to_traditional(self._translator.translate(text, keep))
