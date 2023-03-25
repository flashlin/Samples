import os
import pickle
import numpy as np
from transformers import BertTokenizer


class MyBertTokenizer:
    def __init__(self):
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-cased', never_split=['>='])
        self.EOS = chr(0)
        tokenizer.add_tokens(['"""', '>=', '<=', '!=', '<>', '~=', '+=', '-=', '++', '--', ' ',
                              '\n', '\r', '\t',
                              self.EOS])
        self.EOS_IDX = self.encode(self.EOS)[0]

    def __len__(self):
        return len(self.tokenizer.word_index)

    def tokenize(self, text):
        words = text.split()
        new_words = [word + self.EOS for word in words]
        return new_words

    def fit_on_texts(self, texts):
        pass

    def encode(self, tokens):
        sequences = self.tokenizer.encode(tokens)
        return sequences[1:-1]

    def decode(self, sequence):
        return self.join_words([self.index_word(idx) for idx in sequence if idx != self.EOS_IDX])

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
