import os
import pickle

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

VOCAB_PICKLE = 'vocab.pickle'


class Vocabulary:
    def __init__(self, num_words=9000):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.symbols = {
            ',': '<COMMA>'
        }
        # self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.tokenizer.word_index)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def texts_to_sequences(self, texts):
        return self.tokenizer.texts_to_sequences(texts)

    def index_word(self, idx):
        if idx == 0:
            return None
        if idx >= len(self):
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

    def normal_text(self, text):
        words = text.split()
        words = [self.symbols[word] if word in self.symbols else word for word in words]
        return ' '.join(words)

    def create_n_gram_corpus(self, corpus, max_len=10):
        new_corpus = []
        for text in corpus:
            text = self.normal_text(text)
            sentence = text.split()
            sentence = sentence + ['<EOS>']
            words = sentence
            # 下一個字
            for i in range(3, len(words)):
                new_words = words[: i]
                new_words = self.pad_words(new_words, max_len)
                if self.is_spec_symbol_word(new_words[-1]):
                    continue
                new_text = ' '.join(new_words)
                new_corpus.append(new_text)
            # 克漏字
            for i in range(1, len(words)-1):
                new_words = words[: i] + ['<FILL>'] + words[i+1:] + words[i:i+1]
                new_words = self.pad_words(new_words, max_len)
                if self.is_spec_symbol_word(new_words[-1]):
                    continue
                new_text = ' '.join(new_words)
                new_corpus.append(new_text)
        self.tokenizer.fit_on_texts(new_corpus)
        return new_corpus

    def is_spec_symbol_word(self, word):
        for symbol, mark in self.symbols.items():
            if mark == word:
                return True
        return False

    def create_train_data(self, n_grams_corpus):
        sequences = self.tokenizer.texts_to_sequences(n_grams_corpus)
        # for seq in sequences:
        #     print(f'{seq}')
        sequences = np.array(sequences)
        # 將輸入和輸出分開
        x = sequences[:, :-1]
        y = sequences[:, -1]
        return x, y

    def split_n_grams_to_xy(self, n_grams, n_gram):
        x_grams = []
        y_trues = []
        for words in n_grams:
            x, y = self.split_words_to_xy(words)
            x = self.pad_sequences(x, n_gram)
            x_grams.append(x)
            y_trues.append(y)
        return x_grams, y_trues

    @staticmethod
    def pad_sequences(seq, max_len):
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        return seq

    def split_words_to_xy(self, words):
        x = words[:-1]
        y = words[-1]
        return x, y

    @staticmethod
    def pad_words(words, max_len):
        # if len(words) > max_len:
        #     error_message = f"The length of words must be less than or equal to {max_len}. {words=}"
        #     raise ValueError(error_message)
        if len(words) < max_len:
            words = ['<EOS>'] * (max_len - len(words)) + words
        return words
