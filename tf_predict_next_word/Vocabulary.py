import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_PICKLE = 'vocab.pickle'


class Vocabulary:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.tokenizer.word_index)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def index_word(self, idx):
        if idx == 0:
            return None
        return self.tokenizer.index_word[idx]

    def save(self, vocab_path=VOCAB_PICKLE):
        with open(vocab_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, vocab_path=VOCAB_PICKLE):
        with open(vocab_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def try_load(self, vocab_path=VOCAB_PICKLE):
        if not os.path.exists(vocab_path):
            return False
        print(f'load vocab {vocab_path}')
        self.load(vocab_path)
        return True
