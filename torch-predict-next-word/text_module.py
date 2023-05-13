from typing import Optional

from tsql_tokenizr import tsql_tokenize
import re


def is_start_letter(word: str) -> bool:
    if re.match(r'^[a-zA-Z_]', word):
        return True
    return False


def is_end_letter(word: str) -> bool:
    if re.match(r'[a-zA-Z_0-9]$', word):
        return True
    return False


def is_word(word: str) -> bool:
    if is_start_letter(word) and is_end_letter(word):
        return True
    return False


def create_mask_text(text, offset, mask_len=1) -> Optional[str]:
    tokens = tsql_tokenize(text)
    if offset >= len(tokens):
        return None
    tokens_text = [token.text for token in tokens]
    normal_text = ''.join(tokens_text)
    prev_text = ''.join(tokens_text[: offset])
    curr_tokens = tokens_text[offset: offset + mask_len]
    curr_last_word = curr_tokens[-1].strip()
    if curr_last_word == '':
        return None
    curr_text = ''.join(curr_tokens)
    after_tokens = tokens_text[offset + mask_len:]
    if not after_tokens:
        return None
    after_first_word = after_tokens[0]
    if after_first_word.strip() != '':
        return None
    after_text = ''.join(tokens_text[offset + mask_len:])
    if len(normal_text) == len(curr_text):
        return None
    return prev_text + "<mask>" + after_text + "<eos>" + curr_text


def create_mask_texts(text, mask_len=1):
    mask_texts = []
    for offset in range(0, len(text) - 1):
        mask_text = create_mask_text(text, offset, mask_len)
        if mask_text is None:
            continue
        mask_texts.append(mask_text)
    return mask_texts


def create_all_mask_texts(text):
    all_mask_texts = []
    for mask_len in range(1, len(text) - 1):
        mask_texts = create_mask_texts(text, mask_len)
        all_mask_texts.extend(mask_texts)
    return all_mask_texts
