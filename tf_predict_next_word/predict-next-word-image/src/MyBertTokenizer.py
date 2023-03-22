import os
import pickle
import numpy as np
from transformers import BertTokenizer


class MyBertTokenizer:
    def __init__(self):
        self.tokenizer = tokenizer = BertTokenizer.from_pretrained('bert-base-cased', never_split=['>='])
        tokenizer.add_tokens(['"""', '>=', '<=', '!=', '<>', '~=', '+=', '-=', '++', '--'])

    def __len__(self):
        return len(self.tokenizer.word_index)

    def tokenize(self, text):
        words = text.split()
        if len(words) == 1 and words[0] == '<EOS>':
            return words
        words = words + ['<EOS>']
        return words

    def fit_on_texts(self, texts):
        pass

    def encode(self, tokens):
        # tokens = self.tokenize(text)
        sequences = self.tokenizer.encode(tokens)
        return sequences[1:-1]

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.encode(text)
            sequences.append(sequence)
        return sequences

    def index_word(self, idx):
        text_restored = self.tokenizer.decode([idx])
        return text_restored[1]

    def save(self, vocab_path):
        pass

    def load(self, vocab_path):
        pass

    def try_load(self, vocab_path):
        return True
