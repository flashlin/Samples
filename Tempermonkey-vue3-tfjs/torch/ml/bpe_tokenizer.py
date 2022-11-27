import os
import re

import torch
from functools import lru_cache
from pathlib import Path

"""
https://github.com/lucidrains/DALLE2-pytorch/blob/91c8d1ca1329644d75853fe8992d6aac03b2a992/dalle2_pytorch/tokenizer.py#L18
"""


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/bpe_simple_vocab_16e6.txt")


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + \
         list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def whitespace_clean(text):
    # text = re.sub(r'\s+', ' ', text)
    # text = text.strip()
    return text


def basic_clean(text):
    # text = ftfy.fix_text(text)
    # text = html.unescape(html.unescape(text))
    # text.strip()
    return text


BOS_SYMBOL = '<start_of_text>'
EOS_SYMBOL = '<end_of_text>'
PAD_SYMBOL = '<pad>'
MASK_SYMBOL = '<mask>'
UNK_SYMBOL = '<unk>'


class SimpleTokenizer(object):
    def __init__(self, tokenize_fn, bpe_path=default_bpe()):
        self.tokenize_fn = tokenize_fn
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = Path(bpe_path).read_text(encoding='utf8').split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend([BOS_SYMBOL, EOS_SYMBOL, PAD_SYMBOL, MASK_SYMBOL, UNK_SYMBOL])

        # self.vocab_size = 49408
        self.vocab = vocab
        self.vocab_size, self.encoder, self.decoder = self.calculate_encoder_decoder(vocab)
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {BOS_SYMBOL: BOS_SYMBOL,
                      EOS_SYMBOL: EOS_SYMBOL,
                      PAD_SYMBOL: PAD_SYMBOL,
                      MASK_SYMBOL: MASK_SYMBOL,
                      UNK_SYMBOL: UNK_SYMBOL
                      }
        # self.pattern = re.compile(
        #     r"""<start_of_text>|<end_of_text>""",
        #     re.IGNORECASE)
        self.bos_idx = self.encoder[BOS_SYMBOL]
        self.eos_idx = self.encoder[EOS_SYMBOL]
        self.padding_idx = self.encoder[PAD_SYMBOL]
        self.mask_idx = self.encoder[MASK_SYMBOL]
        self.unk_idx = self.encoder[UNK_SYMBOL]

    def calculate_encoder_decoder(self, vocab):
        encoder = dict(zip(vocab, range(len(vocab))))
        vocab_size = len(encoder)
        decoder = {v: k for k, v in encoder.items()}
        return vocab_size, encoder, decoder

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text, add_start_end=False):
        bpe_tokens = []
        tokens = self.tokenize_fn(text)
        for token in tokens:
            token_text = ''.join(self.byte_encoder[b] for b in token.text.encode_tokens('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token_text).split(' '))
        if add_start_end:
            bpe_tokens = [self.bos_idx] + bpe_tokens + [self.eos_idx]
        return bpe_tokens

    def decode(self, tokens, remove_start_end=True):
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()

        if remove_start_end:
            tokens = [token for token in tokens if token not in (self.bos_idx, self.eos_idx)]
        text = ''.join([self.decoder[token] for token in tokens if token not in [self.padding_idx]])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', '')
        return text

    def tokenize(self, texts, context_length=256, truncate_text=False):
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = [self.encode(text) for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate_text:
                    tokens = tokens[:context_length]
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


if __name__ == '__main__':
    tk = SimpleTokenizer(None)
    if 1 not in [1]:
        print(f"1 not")
    print(f"{tk.bos_idx=} {tk.vocab_size=}")
#     tokens = tk.encode('from tb1 in customer select new{ tb1.name, lastName="flash" }')
#     print(f"{tokens=}")
