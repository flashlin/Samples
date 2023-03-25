import os
import pickle
import numpy as np
from transformers import BertTokenizer


class MyBertTokenizer:
    def __init__(self):
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-cased', never_split=['>='])
        self.EOS = chr(0)
        self.FILL = chr(1)
        self.CAT = chr(2)
        tokenizer.add_tokens([
            self.EOS, self.FILL, self.CAT,
            '"""', '>=', '<=', '!=', '<>', '~=', '+=', '-=', '++', '--', ' ',
            '\n', '\r', '\t'
            ])
        self.EOS_IDX = self.internal_encode(self.EOS)[0]
        self.FILL_IDX = self.internal_encode(self.FILL)[0]
        self.CAT_IDX = self.internal_encode(self.CAT)[0]

    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, text):
        words = text.split()
        return words

    def fit_on_texts(self, texts):
        pass

    def encode(self, tokens):
        sequences = []
        for token in tokens:
            values = self.internal_encode(token)
            values = np.concatenate((values, [self.CAT_IDX]))
            sequences = np.concatenate((sequences, values))
        # print(f'MyBert encode {tokens=} {sequences=}')
        sequences = np.concatenate((sequences[:-1], [self.EOS_IDX]))
        return sequences

    def internal_encode(self, tokens):
        return self.tokenizer.encode(tokens)[1:-1]

    def decode(self, sequence):
        text = ''
        sequence = self.r_trim(sequence)
        for idx in sequence:
            if idx == self.EOS_IDX:
                break
            if idx == self.CAT_IDX:
                text += ' '
                continue
            word = self.index_word(idx)
            if word.startswith('##'):
                text += word[2:]
                continue
            text += word
        return text
        # return self.join_words([self.index_word(idx) for idx in sequence if idx != self.EOS_IDX])

    def r_trim(self, words):
        i = len(words) - 1
        while i >= 0 and words[i] == self.EOS_IDX:
            i -= 1
        return words[:i+1]

    @staticmethod
    def join_words(words):
        text = ''
        for word in words:
            if word.startswith('##'):
                text += word[2:]
            else:
                text += word
        return text

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            tokens = self.tokenize(text)
            sequence = self.encode(tokens)
            sequences.append(sequence)
        return sequences

    def index_word(self, idx):
        text_restored = self.tokenizer.decode([idx])
        return text_restored

    def save(self, vocab_path):
        pass

    def load(self, vocab_path):
        pass

    def try_load(self, vocab_path):
        return True
