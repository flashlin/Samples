import numpy as np
import tensorflow as tf
from model import create_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

number_of_words = 100
batch_size = 64
hidden_size = 512
num_epochs = 100

initial_learning_rate = 0.01
decay_steps = 1000
decay_rate = 0.95
learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

train_data = [
    'select id from customer',
    'select name from customer'
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)
vocab_size = len(tokenizer.word_index)
print(f'{vocab_size=}')

model = create_model(
    total_words=vocab_size + 1, 
    hidden_size=hidden_size,
    num_steps=number_of_words, optimizer=optimizer)
print(model.summary())                

# Convert the target array to one-hot encoded format
# 將文本轉換為數字序列
sequences = tokenizer.texts_to_sequences(train_data)
print(f'{sequences=}')
# 將序列補齊到固定長度
padded_sequences = pad_sequences(sequences, maxlen=number_of_words, padding='post')
one_hot_targets = to_categorical(padded_sequences, num_classes=vocab_size + 1)
# Train the model with the one-hot encoded target array
model.fit(padded_sequences, one_hot_targets, epochs=num_epochs, batch_size=batch_size)


# 預測下一個字
def predict():
    input_text = 'select id from'
    input_sequence = tokenizer.texts_to_sequences([input_text])[0]
    padded_input_sequence = pad_sequences([input_sequence], maxlen=number_of_words, padding='post')
    output_probabilities = model.predict(padded_input_sequence)[0][-1]

    # 將概率值轉換為單詞
    next_word_id = np.argmax(output_probabilities)
    print(f'{next_word_id=}')
    if next_word_id == 0:
        print(f'<EOS>')
        return
    next_word = tokenizer.index_word[next_word_id]
    print(next_word)

predict()    