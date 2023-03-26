import os

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

t = [
    'select id from customer',
    'select id,name from customer',
    'select name from customer'
]

tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

model_path = 'my_model_weights'


def fit(train_data):
    inputs = tokenizer(train_data, padding=True, truncation=True, return_tensors="tf")
    inputs["input_ids"] = tf.where(inputs["input_ids"] == tokenizer.mask_token_id, -100, inputs["input_ids"])
    if os.path.exists(model_path):
        model = TFAutoModelForMaskedLM.from_pretrained(model_path)
    else:
        model = TFAutoModelForMaskedLM.from_pretrained('microsoft/codebert-base')
    print('start training')
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    for epoch in range(100):
        with tf.GradientTape() as tape:
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            loss = tf.reduce_mean(outputs.logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 10 == 0:
            print("Epoch {}, Loss {}".format(epoch, loss))
    model.save_pretrained(model_path)
    return model


def predict(model, text="select [MASK]"):
    print(f'{text=}')
    input_ids = tokenizer.encode(text, return_tensors="tf")
    outputs = model(input_ids)
    predictions = outputs.logits[0, -1]
    top_5 = tf.math.top_k(predictions, k=5)
    for i, idx in enumerate(top_5.indices.numpy()):
        predicted_token = tokenizer.convert_ids_to_tokens([idx])[0]
        print(f"{i + 1}. {predicted_token}")


model = fit(t)
predict(model, 'select [MASK]')
