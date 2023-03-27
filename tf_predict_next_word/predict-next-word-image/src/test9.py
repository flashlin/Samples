import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads=128, ff_dim=128, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads=128, ff_dim=128, rate=0.1):
        super().__init__()
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs):
        attn1 = self.att1(inputs, inputs)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + inputs)

        attn2 = self.att2(out1, out1)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(ffn_output + out2)


class Transformer(keras.Model):
    def __init__(self, num_chars=52, embed_dim=16):
        super().__init__()
        self.embed = layers.Embedding(num_chars, embed_dim)
        self.encoder = TransformerEncoder(embed_dim)
        self.decoder = TransformerDecoder(embed_dim)

    def call(self, inputs):
        x = self.embed(inputs)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode_word(self, word):
        word = tf.strings.unicode_decode(word, 'UTF-8')
        word = tf.reshape(word, (1, -1))
        encoded_word = self(word)
        return encoded_word.numpy()

    def decode_vector(self, vector):
        vector = tf.convert_to_tensor(vector)
        vector = tf.reshape(vector, (1, -1))
        decoded_vector = self(vector)
        decoded_vector = tf.argmax(decoded_vector[0], axis=-1)
        decoded_vector = tf.strings.unicode_encode(decoded_vector, 'UTF-8')
        return decoded_vector.numpy().decode('utf-8')

    def fit(self, X):
        optimizer = Adam(learning_rate=0.00001)
        # self.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
        checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True)
        super().fit(X, X, epochs=100, callbacks=[checkpoint])


model = Transformer()

train_data = ['id', 'name', 'ID', 'Name']
x = []

max_length = max([len(word) for word in train_data])
x = np.zeros((len(train_data), max_length))

for i, word in enumerate(train_data):
    values = [ord(c) for c in word]
    x[i, :len(values)] = values

if os.path.exists('best_model.h5'):
    model.load_weights('best_model.h5')
model.fit(x)
