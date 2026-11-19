import re
import unicodedata
from dataclasses import dataclass

CHINESE_PUNCTUATION = "，。？！、；："
MEANINGFUL_TRAILING_MARKS = "？！"
SENTENCE_END = "。？！.?!"
_TO_ASCII = {"，": ",", "。": ".", "？": "?", "！": "!", "；": ";", "：": ":", "、": ","}
_SENTENCE_SPLIT = re.compile(r"(?<=[。？！!?])|(?<=\.)(?=\s|$)")
_CLAUSE_SPLIT = re.compile(r"(?<=[，,；;：:])")
_CJK_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2CEAF))


@dataclass(frozen=True)
class TimedToken:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def _is_word_char(char: str) -> bool:
    return char == "'" or unicodedata.category(char)[0] in "LN"


def alignment_units(text: str) -> list[str]:
    units: list[str] = []
    word: list[str] = []
    for char in text:
        if _is_cjk(char) or not _is_word_char(char):
            if word:
                units.append("".join(word))
                word = []
            if _is_cjk(char):
                units.append(char)
        else:
            word.append(char)
    if word:
        units.append("".join(word))
    return units


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def _next_text_index(original: str, start: int) -> int | None:
    for index in range(start, len(original)):
        candidate = original[index]
        if not (candidate.isspace() or _is_punctuation(candidate)):
            return index
    return None


def _has_punctuation_at(original: str, start: int) -> bool:
    rest = original[start:].lstrip()
    return bool(rest) and _is_punctuation(rest[0])


def _current_clause(preceding: str) -> str:
    boundary = max(preceding.rfind(mark) for mark in CHINESE_PUNCTUATION + ",.?!;:")
    return preceding[boundary + 1 :]


def _render_mark(mark: str, preceding: str) -> str:
    clause = _current_clause(preceding)
    if clause.strip() and not any(_is_cjk(char) for char in clause):
        return _TO_ASCII.get(mark, mark) + " "
    return mark


def merge_punctuation(original: str, punctuated: str) -> str:
    marks: dict[int, str] = {}
    position = 0
    for char in punctuated:
        if char.isspace():
            continue
        if char in CHINESE_PUNCTUATION:
            if not _has_punctuation_at(original, position):
                marks.setdefault(position, char)
            continue
        found = _next_text_index(original, position)
        if found is not None:
            position = found + 1
    trailing = marks.get(len(original))
    if trailing is not None and trailing not in MEANINGFUL_TRAILING_MARKS:
        del marks[len(original)]
    pieces: list[str] = []
    skip_space = False
    for index in range(len(original) + 1):
        if index in marks:
            rendered = _render_mark(marks[index], "".join(pieces))
            pieces.append(rendered)
            skip_space = not rendered.endswith(" ")
        if index < len(original):
            char = original[index]
            if not (skip_space and char.isspace()):
                pieces.append(char)
                skip_space = False
    return re.sub(r" {2,}", " ", "".join(pieces)).strip()


def split_sentences(text: str, max_units: int) -> list[str]:
    sentences: list[str] = []
    for sentence in (part.strip() for part in _SENTENCE_SPLIT.split(text)):
        if not sentence:
            continue
        if len(alignment_units(sentence)) <= max_units:
            sentences.append(sentence)
            continue
        sentences.extend(_pack_clauses([clause.strip() for clause in _CLAUSE_SPLIT.split(sentence) if clause.strip()], max_units))
    return sentences


def _pack_clauses(clauses: list[str], max_units: int) -> list[str]:
    packed: list[str] = []
    for clause in clauses:
        joiner = " " if packed and packed[-1][-1:].isascii() and clause[:1].isascii() else ""
        if packed and len(alignment_units(packed[-1] + clause)) <= max_units:
            packed[-1] = packed[-1] + joiner + clause
        else:
            packed.append(clause)
    return packed


def time_sentences(sentences: list[str], tokens: list[TimedToken], fallback_duration: float) -> list[Segment]:
    counts = [len(alignment_units(sentence)) for sentence in sentences]
    if tokens and sum(counts) == len(tokens):
        return _time_by_tokens(sentences, counts, tokens)
    return _time_proportionally(sentences, counts, fallback_duration)


def _time_by_tokens(sentences: list[str], counts: list[int], tokens: list[TimedToken]) -> list[Segment]:
    segments: list[Segment] = []
    index = 0
    for sentence, count in zip(sentences, counts):
        if count == 0:
            continue
        first, last = tokens[index], tokens[index + count - 1]
        start = max(first.start, segments[-1].end if segments else 0.0)
        segments.append(Segment(start=start, end=max(start, last.end), text=sentence))
        index += count
    return segments


def _time_proportionally(sentences: list[str], counts: list[int], duration: float) -> list[Segment]:
    total = sum(counts) or 1
    segments: list[Segment] = []
    cursor = 0.0
    for sentence, count in zip(sentences, counts):
        length = duration * max(count, 1) / total
        segments.append(Segment(start=cursor, end=min(duration, cursor + length), text=sentence))
        cursor += length
    return segments
