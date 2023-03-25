import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')


def predict(model, input_text):
    input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
    input_tokens = tf.constant(input_tokens)[None, :]  # add batch dimension
    # predict the next word
    predictions = model(input_tokens)[0][0][-1]
    assert predictions.shape == (tokenizer.vocab_size,), "Invalid shape of predictions tensor"
    top_k_values, top_k_indexes = tf.math.top_k(predictions, k=5)
    predicted_tokens = tokenizer.convert_ids_to_tokens(top_k_indexes.numpy())
    print("Top 5 predictions and their probabilities:")
    for i in range(5):
        print(f"{predicted_tokens[i]}: {top_k_values[i].numpy()}")


def fit(training_data):
    input_ids = [tokenizer.encode(text, add_special_tokens=True) for text in training_data]
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    model = TFBertForMaskedLM.from_pretrained("my_model_weights")
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    model.fit(input_ids, input_ids, epochs=100, batch_size=1)
    model.save_pretrained('my_model_weights')
    return model


training_data = ["select id from customer", "select id,name from prod"]
model = fit(training_data)
# model = TFBertForMaskedLM.from_pretrained("my_model_weights")

predict(model, 'select id from ')
print('worked')