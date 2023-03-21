import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense

# 定義模型
inputs = Input(shape=(10, 1))
lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(inputs)
outputs = Dense(1)(lstm_layer)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')

# 準備訓練資料
data = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 4, 5, 6, 7, 8, 9, 10, 3, 2]
]
y_indices = [
  [0, 1],
  [1, 2, 3]
]
y_values = [33, 1, 2, 3]
y_shape = [2, 3]
y_sparse = tf.sparse.SparseTensor(indices=y_indices, values=y_values, dense_shape=y_shape)

# 訓練模型
model.fit(x=data, y=y_sparse, batch_size=2, epochs=10)
