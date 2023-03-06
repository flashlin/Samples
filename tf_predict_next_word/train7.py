import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from Vocabulary import Vocabulary

corpus = [
    "<EOS>",
    "select id from customer",
    "select id , name from customer"
]

vocab = Vocabulary()
vocab.fit(corpus)

# 建立 nGram 資料集
n_grams = vocab.create_n_gram_corpus(corpus)
# print(f'{n_grams=}')

# 將資料集轉換成 Numpy array
x, y = vocab.create_train_data(n_grams)
# print(f'{x=}')
# print(f'{y=}')

# 建立模型
num_words = len(vocab)
embedding_size = 100
lstm_units = 128
batch_size = 64
epochs = 100
input_length = 10
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=input_length),
    tf.keras.layers.LSTM(lstm_units),
    tf.keras.layers.Dense(num_words, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# x = np.asarray(x, dtype=object)
# y = np.asarray(y, dtype=int)
# x = pad_sequences(x, maxlen=input_length, padding='post')
# y = y.reshape(-1, 1)

print(f'{x=}')
print(f'{y=}')

# 訓練模型
model.fit(x, y, batch_size=batch_size, epochs=epochs)


print(f'{len(vocab)=}')

def predict_next_word(model, tokenizer, test_text, top_k=5):
    # 預處理輸入句子
    test_seq = tokenizer.texts_to_sequences([test_text])[0]
    # test_seq = pad_sequences([test_seq], maxlen=input_length, padding='pre')
    test_seq = pad_sequences([test_seq], maxlen=input_length, padding='post')

    print(f'{test_seq=}')

    # 使用模型預測下一個單詞的機率分佈
    pred_prob = model.predict(test_seq)[0]

    print(f'{pred_prob=}')

    # 取出最高的 k 個機率值和對應的單詞索引
    top_k_idx = pred_prob.argsort()[-top_k:][::-1]
    top_k_prob = pred_prob[top_k_idx]

    # 將單詞索引轉換為對應的單詞
    top_k_word = [tokenizer.index_word(idx+1) for idx in top_k_idx]

    return top_k_word, top_k_prob


def predict(test_text):
    top_k_word, top_k_prob = predict_next_word(model, vocab, test_text, top_k=5)
    print(f"'{test_text}' Top 5 predicted next words and probabilities:")
    for word, prob in zip(top_k_word, top_k_prob):
        print(f"{word}: {prob:.4f}")

predict("select id")

