import os.path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class Word2Vec(keras.Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = layers.Embedding(vocab_size,
                                                 embedding_dim,
                                                 input_length=1,
                                                 name="w2v_embedding")
        self.context_embedding = layers.Embedding(vocab_size,
                                                  embedding_dim,
                                                  input_length=1)

    def call(self, pair):
        target = pair[:, 0]
        context = pair[:, 1]
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dot_product = tf.reduce_sum(tf.multiply(word_emb, context_emb), axis=1)
        return dot_product


def generate_training_data(data, window_size, num_ns):
    data_index = 0
    for target_word in data:
        context_window = np.random.randint(1, window_size + 1)
        for i in range(-context_window, context_window + 1):
            context_word_idx = data_index + i
            if context_word_idx < 0 or context_word_idx >= len(data) or data_index == context_word_idx:
                continue
            context_word = data[context_word_idx]
            yield target_word, context_word
        data_index += 1


def custom_loss(x_logit, y_true):
    # return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
    # return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.cast(y_true, tf.float32))
    x_logit = tf.cast(x_logit, tf.float32)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.cast(y_true, tf.float32)))


def encode(word, word_index):
    return word_index[word]


def decode(code, index_word):
    return index_word[code]


data = ['id', 'name', 'Name']
vocab = set(data)
vocab_size = len(vocab)
word_index = {word: index for index, word in enumerate(vocab)}
index_word = {index: word for index, word in enumerate(vocab)}

window_size = 2
num_ns = 4
embedding_dim = 3

model = Word2Vec(vocab_size, embedding_dim)
model.compile(optimizer='adam', loss=custom_loss)

x_train = []
y_train = []
for target_word, context_word in generate_training_data(data, window_size, num_ns):
    x_train.append([encode(target_word, word_index), encode(context_word, word_index)])
    y_train.append(1)

x_train = np.array(x_train)
y_train = np.array(y_train)

if os.path.exists('word2vec'):
    model.load_weights('word2vec')

epochs = 100
for epoch in range(epochs):
    model.fit(x_train, y_train, batch_size=1)
    if epoch % 10 == 0:
        model.save_weights('word2vec')

encoded_data = [encode(word, word_index) for word in data]
encoded_data = np.array(encoded_data)
output_vectors = model.target_embedding(encoded_data).numpy()
print(output_vectors)

decoded_data = [decode(code, index_word) for code in encoded_data]
print(decoded_data)
