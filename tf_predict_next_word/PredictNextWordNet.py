import os.path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Vocabulary import Vocabulary


class PredictNextWordConfig:
    num_words = 9000
    embedding_size = 100
    lstm_units = 128
    batch_size = 64
    epochs = 100
    input_length = 10
    vocab_file = 'models/predict.vocab'
    model_folder = 'models/predict'


class PredictNextWordModel:
    def __init__(self, config: PredictNextWordConfig):
        self.config = config
        self.vocab = Vocabulary(config.num_words)
        show_epochs = 10
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            config.model_folder, monitor='loss', verbose=1,
            save_best_only=True, mode='min', period=show_epochs)
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.config.num_words,
                                      self.config.embedding_size,
                                      input_length=self.config.input_length - 1),
            tf.keras.layers.LSTM(self.config.lstm_units),
            tf.keras.layers.Dense(self.config.num_words, activation="softmax")
        ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
        return model

    def fit(self, corpus, batch_size, epochs):
        self.vocab.try_load(self.config.vocab_file)
        self.vocab.fit(corpus)
        n_grams = self.vocab.create_n_gram_corpus(corpus)
        self.vocab.save(self.config.vocab_file)
        x, y = self.vocab.create_train_data(n_grams)
        self.try_load_model(self.config.model_folder)
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[self.checkpoint])

    def try_load_model(self, model_folder):
        if not os.path.exists(model_folder):
            return
        print(f'load model from "{model_folder}"')
        self.model = tf.keras.models.load_model(self.config.model_folder)

    def save_model(self, model_file='models/predict.pb'):
        self.model.save(model_file)

    def predict_next_word(self, test_text, top_k=5):
        test_seq = self.vocab.texts_to_sequences([test_text])[0]
        test_seq = pad_sequences([test_seq], maxlen=self.config.input_length - 1, padding='pre')
        print(f'{test_seq=}')
        # 使用模型預測下一個單詞的機率分佈
        pred_prob = self.model.predict(test_seq)[0]
        # 取出最高的 k 個機率值和對應的單詞索引
        top_k_idx = pred_prob.argsort()[-top_k:][::-1]
        top_k_prob = pred_prob[top_k_idx]
        # 將單詞索引轉換為對應的單詞
        top_k_word = [self.vocab.index_word(idx + 1) for idx in top_k_idx]
        return top_k_word, top_k_prob
