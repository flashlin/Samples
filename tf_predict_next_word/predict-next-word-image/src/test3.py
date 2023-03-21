import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, max_length):
        super(MyModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)
        self.lstm = tf.keras.layers.LSTM(64)
        self.dense = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        return self.output_layer(x)


vocab_size = 6
embedding_dim = 16
max_length = 100
model = MyModel(vocab_size, embedding_dim, max_length)

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
texts = ['hello world', 'foo bar', 'hello baz']
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, epochs=10)

oov_word = 'qux'
oov_sequence = tokenizer.texts_to_sequences([oov_word])
oov_padded_sequence = pad_sequences(oov_sequence, maxlen=max_length, padding='post')
oov_vector = model.predict(oov_padded_sequence)
print(oov_vector)
