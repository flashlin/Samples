import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 模型參數
n = 3  # nGram 的 n
num_words = 6   # 只考慮最常出現的 10000 個單字
embedding_size = 100
lstm_units = 128
batch_size = 64
epochs = 100

# 載入語料庫
corpus = [
    "select id from customer",
    "select id from",
    "select id , name from customer"
    "select id ,"
    "select id , name"
]

# 轉換為小寫並切割成單詞
corpus = [sentence.lower().split() for sentence in corpus]

# 建立 Tokenizer 並轉換成序列
tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

# 建立 nGram 資料集
n_grams = []
for seq in sequences:
    for i in range(n, len(seq)):
        n_grams.append(seq[i-n:i+1])

print(f'{sequences=}')
print(f'{n_grams=}')

# 將資料集轉換成 Numpy array
n_grams = np.array(n_grams)

# 將輸入和輸出分開
X = n_grams[:, :-1]
y = n_grams[:, -1]

print(f'{X=}')
print(f'{y=}')

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(num_words, embedding_size, input_length=n),
    tf.keras.layers.LSTM(lstm_units),
    tf.keras.layers.Dense(num_words, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

# 訓練模型
model.fit(X, y, batch_size=batch_size, epochs=epochs)


def predict_next_word(model, tokenizer, n_gram, test_text, top_k=5):
    # 預處理輸入句子
    test_seq = tokenizer.texts_to_sequences([test_text])[0]
    test_seq = pad_sequences([test_seq], maxlen=n_gram, padding='pre')

    # 使用模型預測下一個單詞的機率分佈
    pred_prob = model.predict(test_seq)[0]

    # 取出最高的 k 個機率值和對應的單詞索引
    top_k_idx = pred_prob.argsort()[-top_k:][::-1]
    top_k_prob = pred_prob[top_k_idx]

    # 將單詞索引轉換為對應的單詞
    idx_to_word = dict(map(reversed, tokenizer.word_index.items()))
    top_k_word = [idx_to_word[idx+1] for idx in top_k_idx]

    return top_k_word, top_k_prob


def predict(test_text):
    top_k_word, top_k_prob = predict_next_word(model, tokenizer, n, test_text, top_k=5)
    print(f"'{test_text}' Top 5 predicted next words and probabilities:")
    for word, prob in zip(top_k_word, top_k_prob):
        print(f"{word}: {prob:.4f}")

predict("select id")

