import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, Reshape, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the maximum sequence length and the vocabulary size
max_seq_len = 100
vocab_size = 1000

# Define the input layer
# inputs = Input(shape=(max_seq_len-1,))
# embed = Embedding(vocab_size, 128)(inputs)
# # Reshape the input tensor to 3D shape
# reshape = Reshape((max_seq_len-1, 128))(embed)
# # Define the transformer block
# attn = MultiHeadAttention(8, 16)(reshape, reshape)
# out = Dense(vocab_size, activation='softmax')(attn)
# model = Model(inputs=inputs, outputs=out)

# inputs = Input(shape=(None, max_seq_len-1))
# embed = Embedding(vocab_size, 128)(inputs)
# reshape = Reshape((-1, max_seq_len-1, 128))(embed)
# attn = MultiHeadAttention(8, 16)(reshape, reshape)
# flatten = Flatten()(attn)
# out = Dense(1, activation='sigmoid')(flatten)
# model = Model(inputs=inputs, outputs=out)
# model.compile(loss='binary_crossentropy', optimizer='adam')

inputs = Input(shape=(max_seq_len,))
embed = Embedding(vocab_size, 128)(inputs)
reshape = Reshape((max_seq_len, 128))(embed)
attn = MultiHeadAttention(8, 16)(reshape, reshape)
out = Dense(vocab_size, activation='softmax')(attn)
model = Model(inputs=inputs, outputs=out)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

train_data = [
    "SELECT * FROM Customers WHERE Country='Mexico'"
]


def pad_sequences(seq, max_len, pad_value=0):
    assert len(seq) <= max_len, f"len(seq) {len(seq)} <= {max_len}"
    # seq = np.pad(seq, (0, max_len - len(seq)), mode='constant', constant_values=pad_value)
    pad_width = (max_len - len(seq), 0)  # 在左側填充，因此在元組中反轉填充寬度
    seq = np.pad(seq, pad_width=pad_width, mode='constant', constant_values=[pad_value, 0])
    return seq


def create_n_gram_by_values(sequences, fill, eos, max_len=10):
    new_sequences = []
    for sequence in sequences:
        sequence = np.concatenate((sequence, [eos]))
        # 下一個字
        for i in range(3, len(sequence) + 1):
            new_sequence = sequence[: i]
            new_sequence = pad_sequences(new_sequence, max_len, eos)
            # print(f'{new_words=} {words=} {len(words)=} {i=}')
            new_sequences.append(new_sequence)
        # 克漏字
        for i in range(1, len(sequence) - 2):
            mask = sequence[i:i+1]
            new_sequence = np.concatenate((sequence[: i], [fill], sequence[i + 1:], mask))
            new_sequence = pad_sequences(new_sequence, max_len, eos)
            new_sequences.append(new_sequence)
    return new_sequences


train_words = []
for text in train_data:
    words = text.split(' ')
    train_words.append(words)

x_train = []
y_train = []
n_gram = create_n_gram_by_values(train_words, '<mask>', '<eos>', max_len=max_seq_len)
for gram in n_gram:
    prev = gram[:-1]
    next = gram[-1]
    x_train.append(prev)
    y_train.append([next])

print(f'{x_train}')
print(f'{y_train}')

# for x, y in zip(x_train, y_train):
#     print(f'{x=}')
#     print(f'{y=}')

checkpoint = ModelCheckpoint('best_model.h5', monitor='loss', save_best_only=True)
model.fit(x_train, y_train, epochs=100, callbacks=[checkpoint])
