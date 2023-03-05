import tensorflow as tf
from keras.layers import LSTM
from keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy


class ReshapeLabels(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.reshape(inputs, [-1])


def create_model(total_words,
                 hidden_size=512, num_words_of_sentence=100,
                 word_dim=10, optimizer='RMSprop'):
    total_words = total_words + 1
    """
    :param total_words: 語料庫總共有幾個單字
    :param hidden_size:
    :param num_words_of_sentence: 一句話有多少個單字
    :param word_dim: 一個單字有多少維度
    :param optimizer: adam 則在速度和對超參數的敏感度方面表現更好, RMSprop 對於處理非平穩梯度問題非常有效
    :return:
    """
    model = tf.keras.models.Sequential()

    # Embedding layer / Input layer
    model.add(tf.keras.layers.Embedding(
        total_words, hidden_size, input_length=num_words_of_sentence))

    # 4 LSTM layers
    model.add(LSTM(units=hidden_size,
                   input_shape=(num_words_of_sentence, word_dim), return_sequences=True))

    # Fully Connected layer
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.3, seed=0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512)))
    model.add(tf.keras.layers.Activation('relu'))

    # Output Layer
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(total_words)))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    # model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer=optimizer,
    #               metrics=[CategoricalAccuracy()])
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=optimizer,
    #               metrics=[SparseCategoricalAccuracy()])
    return model
