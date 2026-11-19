import numpy as np
import pytest

from caption_server.text_normalizer import TextLanguage, classify_language, terms_present, to_traditional


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Today we talk about transformers.", TextLanguage.ENGLISH),
        ("這個 API 很 slow", TextLanguage.CHINESE),
        ("今天天氣很好", TextLanguage.CHINESE),
        ("OK", TextLanguage.UNKNOWN),
        ("123 456", TextLanguage.UNKNOWN),
        ("", TextLanguage.UNKNOWN),
    ],
)
def test_classify_language(text: str, expected: TextLanguage) -> None:
    assert classify_language(text) is expected


def test_to_traditional_converts_glyphs_without_rewriting_words() -> None:
    assert to_traditional("这个代码来自数据") == "這個代碼來自數據"


def test_terms_present_returns_surface_forms_including_plurals() -> None:
    text = "Every Token looks at the transformers tokens"
    assert terms_present(text, ["token", "transformer", "attention"]) == ["Token", "tokens", "transformers"]
    assert terms_present("tokenizer only", ["token"]) == []
    assert terms_present("two buses", ["bus"]) == ["buses"]


def test_mixed_sentence_keeps_english_and_skips_translation(pipeline, recognizer, translator) -> None:
    caption = pipeline.transcribe(np.zeros(16_000, dtype=np.float32))
    assert caption.text == "這個是 decoder 的輸出"
    assert caption.language is TextLanguage.CHINESE
    assert caption.translation is None
    assert caption.display == caption.text
    assert translator.calls == []


def test_english_sentence_is_shown_bilingual_with_traditional_translation(pipeline, recognizer, translator) -> None:
    recognizer.next_text = "The transformer reads every token."
    caption = pipeline.transcribe(np.zeros(16_000, dtype=np.float32))
    assert caption.language is TextLanguage.ENGLISH
    assert caption.translation == "譯:The transformer reads every token."
    assert caption.display == "The transformer reads every token.\n譯:The transformer reads every token."
    assert translator.calls == [("The transformer reads every token.", ["transformer", "token"])]


def test_english_sentence_translation_only_when_bilingual_disabled(recognizer, translator) -> None:
    from caption_server.caption_pipeline import CaptionPipeline

    pipeline = CaptionPipeline(recognizer, translator, keep_terms=[], bilingual_english=False)
    caption = pipeline.caption_text("Hello there friends")
    assert caption.display == "譯:Hello there friends"
