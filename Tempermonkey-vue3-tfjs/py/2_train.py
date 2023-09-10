import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Activation
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

from utils.linq_translation_data import load_tfrecord_files
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE

# preparing hyperparameters
src_wordEmbed_dim = 18  # 詞向量維度18
src_max_seq_length = 200  # 句長最大值為 100

tgt_wordEmbed_dim = 27
tgt_max_seq_length = 400  # max length of a sentence (including <SOS> and <EOS>)

# dim of context vector
latent_dim = 256  # LSTM 的內部狀態為 256維的向量

enc_layer_1 = LSTM(latent_dim, return_sequences=True, return_state=True, name="1st_layer_enc_LSTM")
enc_layer_2 = LSTM(latent_dim, return_sequences=True, return_state=True, name="2nd_layer_enc_LSTM")
enc_inputs = Input(shape=(src_max_seq_length, src_wordEmbed_dim))
enc_outputs_1, enc_h1, enc_c1 = enc_layer_1(enc_inputs)
enc_outputs_2, enc_h2, enc_c2 = enc_layer_2(enc_outputs_1)
enc_states = [enc_h1, enc_c1, enc_h2, enc_h2]

# Building a 2-layer LSTM decoder
dec_layer_1 = LSTM(latent_dim, return_sequences=True, return_state=True, name="1st_layer_dec_LSTM")
dec_layer_2 = LSTM(latent_dim, return_sequences=True, return_state=False, name="2nd_layer_dec_LSTM")
dec_dense = Dense(tgt_wordEmbed_dim, activation="softmax")
dec_inputs = Input(shape=(tgt_max_seq_length, tgt_wordEmbed_dim))
dec_outputs_1, dec_h1, dec_c1 = dec_layer_1(dec_inputs, initial_state=[enc_h1, enc_c1])
dec_outputs_2 = dec_layer_2(dec_outputs_1, initial_state=[enc_h2, enc_c2])
dec_outputs_final = dec_dense(dec_outputs_2)

# 算出注意力權重
attention_scores = dot([dec_outputs_2, enc_outputs_2], axes=[2, 2])
attention_weights = Activation("softmax")(attention_scores)
print("attention weights - shape: {}".format(
    attention_weights.shape))  # shape: (None, enc_max_seq_length, dec_max_seq_length)

# 利用加權平均求出context vector
context_vec = dot([attention_weights, enc_outputs_2], axes=[2, 1])
print("context vector - shape: {}".format(context_vec.shape))  # shape: (None, dec_max_seq_length, latent_dim)

# 將解碼器當下的內部狀態與context vector連接起來，並得到注意力層的輸出
# concatenate context vector and decoder hidden state h_t
ht_context_vec = concatenate([context_vec, dec_outputs_2], name="concatentated_vector")
print("ht_context_vec - shape: {}".format(ht_context_vec.shape))  # shape: (None, dec_max_seq_length, 2 * latent_dim)

# obtain attentional vector
attention_vec = Dense(latent_dim, use_bias=False, activation="tanh", name="attentional_vector")(ht_context_vec)
print("attention_vec - shape: {}".format(attention_vec.shape))  # shape: (None, dec_max_seq_length, latent_dim)

# 傳入softmax層預估當下輸出值的條件機率, 將注意力機制的輸出值傳入 softmax 層得到當下的目標詞向量
dec_outputs_final = Dense(tgt_wordEmbed_dim, use_bias=False, activation="softmax")(attention_vec)
print("dec_outputs_final - shape: {}".format(
    dec_outputs_final.shape))  # shape: (None, dec_max_seq_length, tgt_wordEmbed_dim)

# Integrate seq2seq model with attention mechanism
model = Model([enc_inputs, dec_inputs], dec_outputs_final, name="seq2seq_2_layers_attention")
model.summary()

# Preview model architecture
#plot_model(model, to_file="output/2-layer_seq2seq_attention.png",
#           dpi=100, show_shapes=True, show_layer_names=True)


# increase 1 dimension
# with np.load('./output/linq-translation_padded.npz') as npz:
#     src_sentences_padded = npz["x"]
#     tgt_sentences_padded = npz["y"]
# src_sentences_padded = src_sentences_padded.reshape(*src_sentences_padded.shape, 1) # shape: (26388, 38, 1)
# tgt_sentences_padded = tgt_sentences_padded.reshape(*tgt_sentences_padded.shape, 1)

# prepare training and test data
data = np.load("./output/linq-translation_padded.npz")
print(data.files)

X = data["x"]
y = data["y"]

# text_as_int = enc_inputs
# characters = tf.data.Dataset.from_tensor_slices(text_as_int)
# print(f"{characters=}")


def one_hot_encode_labels(sequences, vocab_size):
    y_list = []
    for seq in sequences:
        # one-hot encode each sentence
        oh_encoded = to_categorical(seq, num_classes = vocab_size)
        y_list.append(oh_encoded)
    y = np.array(y_list, dtype = np.float32)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

tgt_vocab_size = TSQL_VOCAB_SIZE
dec_outputs = one_hot_encode_labels(y, tgt_vocab_size) # shape: (n_samples, tgt_max_seq_length, tgt_vocab_size)

# test_ratio = .2
# enc_inputs_train, enc_inputs_test = train_test_split(enc_inputs, test_size = test_ratio, shuffle = False)
# # dec_inputs_train, dec_inputs_test = train_test_split(dec_inputs, test_size = test_ratio, shuffle = False)
# y_train, y_test = train_test_split(dec_outputs, test_size = test_ratio, shuffle = False)
# X_train = [enc_inputs_train, dec_inputs_train]
# X_test = [enc_inputs_test, dec_inputs_test]

# train_dataset, valid_dataset, test_dataset = load_tfrecord_files("./output")

# 以下函數允許模型在每個 epoch 運行時更改學習率。
# 當模型沒有改進時，我們可以使用回調來停止訓練。在訓練過程結束時，模型將恢復其最佳迭代的權重
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=20, decay_rate=0.96, staircase=True
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "./output/linq_model.h5", save_best_only=True
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True
)

# ----------------------------
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#     # loss="binary_crossentropy",
#     loss = tf.keras.losses.CategoricalCrossentropy(),
#     metrics=tf.keras.metrics.AUC(name="auc"),
# )
#
# history = model.fit(
#     train_dataset,
#     epochs=2,
#     validation_data=valid_dataset,
#     callbacks=[checkpoint_cb, early_stopping_cb],
# )
#
# print(f"{history=}")