import numpy as np
import pytest

from caption_server.sentence_segmenter import TimedToken
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


def test_mixed_sentence_keeps_english_and_skips_translation(pipeline, translator) -> None:
    chunk = pipeline.transcribe(np.zeros(16_000, dtype=np.float32))
    assert chunk.caption.text == "這個是 decoder 的輸出"
    assert [segment.caption.display for segment in chunk.segments] == ["這個是 decoder 的輸出"]
    assert chunk.segments[0].caption.language is TextLanguage.CHINESE
    assert translator.calls == []


def test_english_sentence_is_shown_bilingual_with_traditional_translation(pipeline, recognizer, translator) -> None:
    recognizer.next_text = "The transformer reads every token."
    chunk = pipeline.transcribe(np.zeros(16_000, dtype=np.float32))
    caption = chunk.segments[0].caption
    assert caption.language is TextLanguage.ENGLISH
    assert caption.translation == "譯:The transformer reads every token."
    assert caption.display == "The transformer reads every token.\n譯:The transformer reads every token."
    assert translator.calls == [("The transformer reads every token.", ["transformer", "token"])]


def test_run_on_text_is_punctuated_split_and_timed_per_sentence(pipeline, recognizer, punctuator, translator) -> None:
    recognizer.next_text = "三千還少這事兒您知道嗎"
    recognizer.next_tokens = [TimedToken(char, index * 0.5, index * 0.5 + 0.4) for index, char in enumerate("三千還少這事兒您知道嗎")]
    punctuator.next_output = "三千還少。這事兒您知道嗎？"
    chunk = pipeline.transcribe(np.zeros(16_000 * 6, dtype=np.float32))
    assert [(segment.start, segment.end, segment.caption.display) for segment in chunk.segments] == [
        (0.0, 1.9, "三千還少。"),
        (2.0, 5.4, "這事兒您知道嗎？"),
    ]
    assert chunk.caption.display == "三千還少。\n這事兒您知道嗎？"


def test_interim_skips_alignment_and_translation(pipeline, recognizer, translator) -> None:
    recognizer.next_text = "Today we talk about attention."
    chunk = pipeline.transcribe(np.zeros(16_000 * 2, dtype=np.float32), final=False)
    assert recognizer.received[-1][1] is False
    assert chunk.segments[0].caption.display == "Today we talk about attention."
    assert translator.calls == []


def test_caption_text_translation_only_when_bilingual_disabled(recognizer, punctuator, translator) -> None:
    from caption_server.caption_pipeline import CaptionPipeline, PipelineOptions

    pipeline = CaptionPipeline(recognizer, punctuator, translator, PipelineOptions(keep_terms=[], bilingual_english=False, max_sentence_units=24))
    assert pipeline.caption_text("Hello there friends").display == "譯:Hello there friends"
