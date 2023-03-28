import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 設置超參數
vocab_size = 128  # ASCII 編碼字符集大小
embedding_dim = 256
rnn_units = 1024
batch_size = 64

# 建立模型
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embedding_dim,
                         batch_input_shape=[batch_size, None]),
        layers.LSTM(rnn_units,
                    return_sequences=True,
                    stateful=True,
                    recurrent_initializer='glorot_uniform'),
        layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)

# 定義損失函數
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# 編譯模型
model.compile(optimizer='adam', loss=loss, run_eagerly=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# 準備訓練數據
seq_length = 100
buffer_size = 10

text = open('input.txt', 'rb').read().decode(encoding='utf-8')
lines = text.split('\n')
sequences = []
for text in lines:
    text_as_int = np.array([ord(c) for c in text])
    sequences.append(text_as_int)

# 將輸入列表轉換為 tf.RaggedTensor
print(f'{sequences=}')
ragged_tensor = tf.ragged.constant(sequences)
# 打印 tf.RaggedTensor 的形狀和值
print("Shape of RaggedTensor:", ragged_tensor.shape)
print("Values of RaggedTensor:", ragged_tensor)
dataset = tf.data.Dataset.from_tensor_slices(ragged_tensor)

# 設置 batch size
# text_as_int = np.array([ord(c) for c in text])
# data = tf.data.Dataset.from_tensor_slices(text_as_int)
# sequences = data.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# 訓練模型
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS)
