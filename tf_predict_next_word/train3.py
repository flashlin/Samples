import os.path

import numpy as np
import tensorflow as tf
from Vocabulary import Vocabulary
from model import create_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

train_data = [
    'select id from customer',
    'select name from customer',
    'select username from customer',
    'select birth from customer',
    'select address from customer',
    'select firstName from customer',
    'select lastName from customer',
    'select email from customer',
    'select mobile from customer',
    'select password from customer',
]

vob_file = 'Models/vocab.pickle'
vocab = Vocabulary()
vocab.try_load(vob_file)
seq1 = vocab.texts_to_sequences(["select id from customer"])
print(f'{seq1=}')

vocab.fit(train_data)
vocab.save(vob_file)
seq2 = vocab.texts_to_sequences(["select password from customer"])
print(f'{seq2=}')


BEST_PREDICT_MODEL = "Models/best_predict_model.hdf5"


class PredictNextWord:
    """
        number_of_words: 關注幾個字?
    """
    def __init__(self, vocab: Vocabulary, number_of_words=100, hidden_size=512 * 2):
        self.vocab = vocab
        self.number_of_words = number_of_words
        self.vocab_size = vocab_size = len(vocab)
        initial_learning_rate = 0.01
        decay_steps = 1000
        decay_rate = 0.95
        learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        self.model = create_model(
            total_words=vocab_size + 1,
            hidden_size=hidden_size,
            num_steps=number_of_words, optimizer=optimizer)
        #print(model.summary())

        show_epochs = 10
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            BEST_PREDICT_MODEL, monitor='val_loss', verbose=1,
            save_best_only=True, mode='min', period=show_epochs)

    def fit(self, texts, num_epochs=10, batch_size=64):
        if os.path.exists(BEST_PREDICT_MODEL):
            self.model.load_weights(BEST_PREDICT_MODEL)
        train_texts, test_texts = train_test_split(texts, train_size=0.7, random_state=42)
        train_padded_sequences = self.pad_texts_to_sequences(train_texts)
        train_one_hot_targets = self.one_hot(train_padded_sequences)
        test_padded_sequences = self.pad_texts_to_sequences(test_texts)
        test_one_hot_targets = self.one_hot(test_padded_sequences)
        self.model.fit(train_padded_sequences, train_one_hot_targets,
                       validation_data=(test_padded_sequences, test_one_hot_targets),
                       epochs=num_epochs, batch_size=batch_size,
                       callbacks=[self.checkpoint])

    def pad_texts_to_sequences(self, texts):
        # 將文本轉換為數字序列
        sequences = self.vocab.texts_to_sequences(texts)
        # 將序列補齊到固定長度
        padded_sequences = pad_sequences(sequences, maxlen=self.number_of_words, padding='post')
        return padded_sequences

    def one_hot(self, padded_sequences):
        one_hot_targets = to_categorical(padded_sequences, num_classes=self.vocab_size + 1)
        return one_hot_targets


    def predict1(self, input_text):
        input_sequence = self.vocab.texts_to_sequences([input_text])[0]
        padded_input_sequence = pad_sequences([input_sequence], maxlen=self.number_of_words, padding='post')
        output_probabilities = self.model.predict(padded_input_sequence)[0][-1]
        # 將概率值轉換為單詞
        next_word_id = np.argmax(output_probabilities)
        if next_word_id == 0:
            return None
        next_word = self.vocab.index_word(next_word_id)
        return next_word

    def predict(self, input_text, top=5):
        input_sequence = self.vocab.texts_to_sequences([input_text])[0]
        padded_input_sequence = pad_sequences([input_sequence], maxlen=self.number_of_words, padding='post')
        output_probabilities = self.model.predict(padded_input_sequence)[0][-1]

        # 取得 TOP 5 概率值的索引
        top_n_indices = np.argsort(output_probabilities)[-top:]
        # 將這些索引轉換為單詞
        top_n_words = [self.vocab.index_word(idx) for idx in top_n_indices]
        # 取得對應的概率值
        top_n_probabilities = np.flip(np.sort(output_probabilities))[:top]
        # 配對單詞和概率值
        word_prob_dict = dict(zip(top_n_words, top_n_probabilities))
        # return top_n_words
        return word_prob_dict

    def save(self, model_path='Models/predict.pb'):
        self.model.save(model_path)

    def load(self, model_path='Models/predict.pb'):
        self.model = tf.keras.models.load_model(model_path)


p = PredictNextWord(vocab, number_of_words=10)
p.fit(train_data, num_epochs=500, batch_size=10)

top_words = p.predict("select id")
# 輸出 TOP 5 預測字
print('Top 5 predicted words:')
for (word, prob) in top_words.items():
    print(f'{word} {prob}')

