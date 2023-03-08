import tensorflow as tf
from transformers import TFBertForMaskedLM, BertTokenizer
from transformers import TFBertModel, BertTokenizer

# model_name = 'bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(model_name)
# BERT = TFBertModel.from_pretrained(model_name)

class BERTMaskedLM(tf.Module):
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForMaskedLM.from_pretrained(model_name)

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors='tf')
        mask_token_index = tf.where(inputs['input_ids'] == self.tokenizer.mask_token_id)
        token_logits = self.model(inputs)['logits']
        mask_token_logits = token_logits[0, mask_token_index[0, 1], :]
        mask_token_probs = tf.nn.softmax(mask_token_logits)
        predicted_index = tf.argmax(mask_token_probs, axis=-1)
        predicted_token = self.tokenizer.decode([predicted_index])
        return predicted_token


# class NextCharPredictor(tf.keras.Model):
#     def __init__(self, num_chars, embedding_size, num_heads, dense_units):
#         super().__init__()
#         self.embedding_layer = BERT(num_heads=num_heads, embedding_size=embedding_size)
#         self.dense_layer = tf.keras.layers.Dense(units=dense_units, activation='relu')
#         self.output_layer = tf.keras.layers.Dense(units=num_chars, activation='softmax')

#     def call(self, inputs):
#         embedding = self.embedding_layer(inputs)
#         dense = self.dense_layer(embedding[:, -1, :])
#         return self.output_layer(dense)


def loss_fn(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)




# Create a BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# input_text = "This is a test sentence."
#input_tokens = tokenizer.tokenize(input_text)
# input_ids = tokenizer.convert_tokens_to_ids(input_tokens)


# 預測
model = BERTMaskedLM()
input_text = "select id from"
predicted_token = model(input_text)
print(predicted_token)


def fine_tune():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForMaskedLM.from_pretrained('bert-base-uncased')

    # Load your training data and convert it to input IDs
    train_data = [
        "123",
        "12323"
    ]
    train_ids = tokenizer(train_data, padding=True, truncation=True, return_tensors='tf')['input_ids']

    # Lock the weights of some layers
    for layer in model.layers[:4]:
        layer.trainable = False
    # Compile the model with appropriate optimizer and loss
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-5), loss='sparse_categorical_crossentropy')
    # Train the model for a few epochs
    model.fit(train_ids, train_ids, epochs=5, batch_size=32)
    # Save the trained model
    model.save_pretrained('path/to/save/dir')
    loaded_model = tf.keras.models.load_model('path/to/saved/model')



#model = NextCharPredictor(num_chars=vocab_size, embedding_size=64, num_heads=4, dense_units=256)
#model.compile(optimizer='adam', loss=loss_fn)


