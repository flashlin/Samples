import tensorflow as tf
from transformers import BertTokenizer, TFBertForMaskedLM

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

input_text = "select ID from"
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

print('worked')