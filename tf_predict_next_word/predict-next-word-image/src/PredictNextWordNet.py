import os.path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from Vocabulary import Vocabulary
from SimpleTokenizer import SimpleTokenizer
from MyBertTokenizer import MyBertTokenizer


class PredictNextWordConfig:
    num_words = 9000
    embedding_size = 100
    lstm_units = 128
    batch_size = 64
    epochs = 20
    input_length = 100
    vocab_file = 'models/predict.vocab'
    model_file = 'models/predict.h5'


class PredictNextWordModel:
    def __init__(self, config: PredictNextWordConfig):
        self.config = config
        self.tokenizer = MyBertTokenizer()
        # self.vocab = Vocabulary(SimpleTokenizer(config.num_words))
        self.vocab = Vocabulary(self.tokenizer)
        self.vocab.try_load(self.config.vocab_file)
        show_epochs = 10
        self.checkpoint = tf.keras.callbacks.ModelCheckpoint(
            config.model_file, monitor='loss', verbose=1,
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
        self.vocab.fit(corpus)
        n_grams = self.vocab.create_n_gram_by_corpus(corpus, self.config.input_length)
        # for sequence in n_grams:
        #     text = self.vocab.tokenizer.decode(sequence)
        #     print(f'n_grams={text}')
        self.vocab.save(self.config.vocab_file)
        x, y = self.vocab.create_train_data(n_grams)
        self.try_load_model(self.config.model_file)
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=[self.checkpoint])
        # self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
        # self.save_model(self.config.model_file)

    def try_load_model(self, model_folder):
        if not os.path.exists(model_folder):
            return
        print(f'load model from "{model_folder}"')
        self.model = tf.keras.models.load_model(self.config.model_file)

    def save_model(self, model_file='models/predict.pb'):
        self.model.save(model_file)

    def predict_next_value(self, test_text, top_k=5):
        eos_id = self.tokenizer.EOS_IDX
        test_seq = self.vocab.texts_to_sequences([test_text])[0]
        print(f'{test_seq=}')
        test_seq = pad_sequences([test_seq], maxlen=self.config.input_length - 1, padding='pre', value=eos_id)
        # 使用模型預測下一個單詞的機率分佈
        pred_prob = self.model.predict(test_seq)[0]
        # 取出最高的 k 個機率值和對應的單詞索引
        top_k_idx_list = pred_prob.argsort()[-top_k:][::-1]
        top_k_prob = pred_prob[top_k_idx_list]
        return top_k_idx_list, top_k_prob

    def predict_next_word(self, test_text, top_k=5):
        def predict(new_text, top_k, prev_idx_list):
            if top_k == self.tokenizer.EOS_IDX:
                return prev_idx_list
            new_text += self.tokenizer.decode([top_k])
            prev_idx_list.append(top_k)
            next = self.predict_next_value(new_text, 1)
            return predict(new_text, next[0].top_k, prev_idx_list)

        top_k_idx, top_k_prob = self.predict_next_value(test_text, top_k)

        top_k_text = []
        for _ in top_k_idx:
            top_k_text.append(test_text)

        top_k_word = []
        for new_text, top_k in zip(top_k_text, top_k_idx):
            top_k_idx_list = predict(new_text, top_k, [top_k])
            word = self.tokenizer.decode(top_k_idx_list)
            top_k_word.append(word)

        # 將單詞索引轉換為對應的單詞
        top_k_word = [self.vocab.index_word(idx) for idx in top_k_idx]
        return top_k_word, top_k_prob
