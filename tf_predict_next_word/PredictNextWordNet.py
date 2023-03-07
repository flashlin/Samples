import os.path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Vocabulary import Vocabulary
from tensorflow.keras.layers import Layer

class MemoryAwareSynapses(Layer):
    def __init__(self, num_synapses, **kwargs):
        super(MemoryAwareSynapses, self).__init__(**kwargs)
        self.num_synapses = num_synapses

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.num_synapses),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, inputs):
        # 計算注意力權重
        attentions = tf.matmul(inputs, self.kernel)
        attentions = tf.nn.softmax(attentions, axis=1)

        # 加權平均 Embedding 向量
        outputs = tf.reduce_sum(tf.multiply(inputs, attentions), axis=1)

        return outputs

class MyModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.config = config
        self.embedding = tf.keras.layers.Embedding(self.config.num_words,
                                      self.config.embedding_size,
                                      input_length=self.config.input_length - 1)
        self.memory_aware = MemoryAwareSynapses(self.config.num_words)
        self.lstm = tf.keras.layers.LSTM(self.config.lstm_units)
        self.dense = tf.keras.layers.Dense(self.config.num_words, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.memory_aware(x)
        x = self.lstm(x)
        x = self.dense(x)
        return x

# model = MyModel(config)
# model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")




class PredictNextWordConfig:
    num_words = 9000
    embedding_size = 100
    lstm_units = 128
    batch_size = 64
    epochs = 100
    input_length = 10
    vocab_file = 'models/predict.vocab'
    model_file = 'models/predict.h5'


class PredictNextWordModel:
    def __init__(self, config: PredictNextWordConfig):
        self.config = config
        self.vocab = Vocabulary(config.num_words)
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
        n_grams = self.vocab.create_n_gram_corpus(corpus)
        # for text in n_grams:
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

    def predict_next_word(self, test_text, top_k=5):
        eos_id = self.vocab.texts_to_sequences(['<EOS>'])
        test_seq = self.vocab.texts_to_sequences([test_text])[0]
        test_seq = pad_sequences([test_seq], maxlen=self.config.input_length - 1, padding='pre', value=eos_id)
        # 使用模型預測下一個單詞的機率分佈
        pred_prob = self.model.predict(test_seq)[0]
        # 取出最高的 k 個機率值和對應的單詞索引
        top_k_idx_list = pred_prob.argsort()[-top_k:][::-1]
        top_k_prob = pred_prob[top_k_idx_list]
        # 將單詞索引轉換為對應的單詞
        top_k_word = [self.vocab.index_word(idx) for idx in top_k_idx_list]
        return top_k_word, top_k_prob
