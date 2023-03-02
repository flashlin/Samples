import tensorflow as tf
from collections import Counter
import numpy as np


class NGramModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.linear1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.linear2 = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        embeds = self.embedding(inputs)
        embeds = tf.reshape(embeds, (1, -1))
        out = self.linear1(embeds)
        out = self.linear2(out)
        log_probs = tf.nn.log_softmax(out, axis=1)
        return log_probs
    
    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        super(NGramModel, self).build(input_shape)


def train(model, loss_function, optimizer, context, target):
    with tf.GradientTape() as tape:
        log_probs = model(context)
        loss = loss_function(target, log_probs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy()

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = tf.constant(idxs, dtype=tf.int64)
    return tensor


def get_data(text, n):
    # Tokenize text
    tokens = text.split()
    # Count the frequency of each word
    word_freq = Counter(tokens)
    # Get a list of unique words
    vocab = sorted(set(tokens))
    # Create a mapping from words to indices
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    data = []
    for i in range(n, len(tokens)):
        context = tokens[i - n:i]
        target = tokens[i]
        # Ignore unknown words
        if target not in word_to_ix:
            continue
        context_vector = make_context_vector(context, word_to_ix)
        target_index = tf.constant([word_to_ix[target]], dtype=tf.int64)
        data.append((context_vector, target_index))
    return data, word_to_ix

model_path = 'models/ngram'

def train_ngram_model(text, n, embedding_dim, epochs, learning_rate):
    # Get training data and vocabulary
    training_data, word_to_ix = get_data(text, n)
    vocab_size = len(word_to_ix)
    # Create a model and optimizer
    model = NGramModel(vocab_size, embedding_dim, n)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # Train the model
    for epoch in range(epochs):
        total_loss = 0
        for context, target in training_data:
            loss = train(model, loss_function, optimizer, context, target)
            total_loss += loss
        print("Epoch %d: loss=%.4f" % (epoch + 1, total_loss))
        model.save(model_path)
    return model, word_to_ix

def predict_next_word(model, word_to_ix, context):
    context_vector = make_context_vector(context, word_to_ix)
    log_probs = model(context_vector)
    top_k = tf.math.top_k(log_probs, k=3)
    predicted_index = top_k.indices.numpy().flatten()
    predicted_words = [list(word_to_ix.keys())[i] for i in predicted_index]
    return predicted_words


text = 'select id from customer <EOS>'
n = 10
embedding_dim = 10
epochs = 100
learning_rate = 0.1

model, word_to_ix = train_ngram_model(text, n, embedding_dim, epochs, learning_rate)

#預測下一個單詞
context = ['select', 'id']
predicted_words = predict_next_word(model, word_to_ix, context)
print(predicted_words) # 預期輸出：['TEXT', 'COMMENTS', 'PROJECT']