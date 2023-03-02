import tensorflow as tf
from model import create_model

number_of_words = 3
batch_size = 200
hidden_size = 1500
num_epochs = 80

initial_learning_rate = 0.01
decay_steps = 1000
decay_rate = 0.95
learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)

total_words = 100

model = create_model(
    total_words=total_words, 
    hidden_size=hidden_size,
    num_steps=number_of_words, optimizer=optimizer)
                
print(model.summary())                