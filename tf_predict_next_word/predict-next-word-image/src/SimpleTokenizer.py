import os
import pickle

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


class SimpleTokenizer:
    def __init__(self, num_words=9000):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>",
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    def __len__(self):
        return len(self.tokenizer.word_index)

    def tokenize(self, text):
        words = text.split()
        # if len(words) == 1 and words[0] == '<EOS>':
        #     return words
        # words = words + ['<EOS>']
        return words

    def fit_on_texts(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def encode(self, text):
        sequence = self.texts_to_sequences([text])
        return np.array(sequence).flatten()

    def decode(self, sequence):
        return ' '.join([self.index_word(idx) for idx in sequence])

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def index_word(self, idx):
        if idx == 0:
            return None
        if idx >= len(self):
            return None
        return self.tokenizer.index_word[idx]

    def save(self, vocab_path):
        with open(vocab_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, vocab_path):
        with open(vocab_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def try_load(self, vocab_path):
        if not os.path.exists(vocab_path):
            return False
        print(f'load vocab {vocab_path}')
        self.load(vocab_path)
        return True
