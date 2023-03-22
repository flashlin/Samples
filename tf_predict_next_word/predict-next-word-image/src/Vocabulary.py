import numpy as np

from SimpleTokenizer import SimpleTokenizer

VOCAB_PICKLE = 'vocab.pickle'


class Vocabulary:
    def __init__(self, tokenizer=None):
        self.tokenizer = SimpleTokenizer() if tokenizer is None else tokenizer

    def __len__(self):
        return len(self.tokenizer)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def try_load(self, vocab_path):
        return self.tokenizer.try_load(vocab_path)

    def save(self, vocab_path=VOCAB_PICKLE):
        self.tokenizer.save(vocab_path)

    def create_n_gram_corpus(self, corpus, max_len=10):
        new_corpus = []
        for text in corpus:
            words = self.tokenizer.tokenize(text)
            # 下一個字
            for i in range(3, len(words) + 1):
                new_words = words[: i]
                new_words = self.pad_words(new_words, max_len)
                new_text = ' '.join(new_words)
                # print(f'{new_text=}')
                new_corpus.append(new_text)
            # 克漏字
            for i in range(1, len(words)-1):
                new_words = words[: i] + ['<FILL>'] + words[i+1:] + words[i:i+1]
                new_words = self.pad_words(new_words, max_len)
                new_text = ' '.join(new_words)
                new_corpus.append(new_text)
        self.tokenizer.fit_on_texts(new_corpus)
        return new_corpus

    def index_word(self, index):
        return self.tokenizer.index_word(index)

    def create_train_data(self, n_grams_corpus):
        sequences = self.texts_to_sequences(n_grams_corpus)
        #print(f'{sequences=}')
        #for seq in sequences:
        #    print(f'{seq}')
        sequences = np.array(sequences)
        # 將輸入和輸出分開
        x = sequences[:, :-1]
        y = sequences[:, -1]
        return x, y

    def texts_to_sequences(self, texts):
        sequences = []
        max_pad_len = 0
        for text in texts:
            words = self.tokenizer.tokenize(text)
            sequence = self.tokenizer.encode(words)
            print(f'{text=} {words=} {sequence=}')
            max_pad_len = max(len(sequence), max_pad_len)
            sequences.append(sequence)
        new_sequences = []
        for sequence in sequences:
            sequence = self.pad_sequences(sequence, max_pad_len)
            new_sequences.append(sequence)
        return new_sequences

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
        assert len(seq) <= max_len
        seq = np.pad(seq, (0, max_len - len(seq)), mode='constant')
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
