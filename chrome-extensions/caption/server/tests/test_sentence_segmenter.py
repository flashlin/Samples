import pytest

from caption_server.sentence_segmenter import Segment, TimedToken, alignment_units, merge_punctuation, split_sentences, time_sentences


@pytest.mark.parametrize(
    ("original", "punctuated", "expected"),
    [
        (
            "接下來輪到 decoder 產生語音辨識的結果那 decoder 怎麼產生這個語音辨識的結果呢",
            "接下來輪到decoder產生語音辨識的結果，那decoder怎麼產生這個語音辨識的結果呢？",
            "接下來輪到 decoder 產生語音辨識的結果，那 decoder 怎麼產生這個語音辨識的結果呢？",
        ),
        (
            "today we are going to talk about transformers the attention mechanism works",
            "today，we are going to talk about transformers，the attention mechanism works。",
            "today, we are going to talk about transformers, the attention mechanism works",
        ),
        ("已經有標點，不要重複。", "已經有標點，不要重複。", "已經有標點，不要重複。"),
        ("Already punctuated.", "Already punctuated。", "Already punctuated."),
        ("模型改了字也不會亂掉", "模型改了子也，不會亂掉。", "模型改了字也，不會亂掉"),
        ("做滿三年", "做滿三年。", "做滿三年"),
        ("我們介紹 encoder 跟 decoder 好嗎", "我們介紹encoder跟decoder，好嗎？", "我們介紹 encoder 跟 decoder，好嗎？"),
        ("先講中文 then switch to English 好", "先講中文，then switch to English。好。", "先講中文，then switch to English. 好"),
        ("", "", ""),
    ],
)
def test_merge_punctuation_keeps_original_text(original: str, punctuated: str, expected: str) -> None:
    assert merge_punctuation(original, punctuated) == expected


def test_alignment_units_match_the_aligner_tokenization() -> None:
    assert alignment_units("輪到 decoder 產生，one-hot's 2 個") == ["輪", "到", "decoder", "產", "生", "one", "hot's", "2", "個"]


def test_split_sentences_on_sentence_marks_and_long_clauses() -> None:
    assert split_sentences("第一句。第二句？Third one. Fourth!", 30) == ["第一句。", "第二句？", "Third one.", "Fourth!"]
    long_sentence = "一二三四五，六七八九十，甲乙丙丁戊。"
    assert split_sentences(long_sentence, 10) == ["一二三四五，六七八九十，", "甲乙丙丁戊。"]
    assert split_sentences("no punctuation at all", 30) == ["no punctuation at all"]


def test_time_sentences_uses_aligned_token_times() -> None:
    tokens = [TimedToken(c, i * 0.5, i * 0.5 + 0.4) for i, c in enumerate("第一句第二句")]
    assert time_sentences(["第一句。", "第二句。"], tokens, 6.0) == [Segment(0.0, 1.4, "第一句。"), Segment(1.5, 2.9, "第二句。")]


def test_time_sentences_falls_back_to_proportional_timing_when_tokens_mismatch() -> None:
    segments = time_sentences(["一二", "三四五六"], [TimedToken("一", 0, 1)], 6.0)
    assert segments == [Segment(0.0, 2.0, "一二"), Segment(2.0, 6.0, "三四五六")]
