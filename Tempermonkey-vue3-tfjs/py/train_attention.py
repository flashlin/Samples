import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot
from tensorflow.keras.layers import Activation

# preparing hyperparameters

# source language- English
src_wordEmbed_dim = 18  # 詞向量維度18
src_max_seq_length = 100  # 句長最大值為 100

# target language- Spanish
tgt_wordEmbed_dim = 27  # dim of text vector representation
tgt_max_seq_length = 12  # max length of a sentence (including <SOS> and <EOS>)

# dim of context vector
latent_dim = 256  # LSTM 的內部狀態為 256維的向量


# Building a 2-layer LSTM encoder
enc_layer_1 = LSTM(latent_dim, return_sequences=True,
                   return_state=True, name="1st_layer_enc_LSTM")
enc_layer_2 = LSTM(latent_dim, return_sequences=True,
                   return_state=True, name="2nd_layer_enc_LSTM")
enc_inputs = Input(shape=(src_max_seq_length, src_wordEmbed_dim))
enc_outputs_1, enc_h1, enc_c1 = enc_layer_1(enc_inputs)
enc_outputs_2, enc_h2, enc_c2 = enc_layer_2(enc_outputs_1)
enc_states = [enc_h1, enc_c1, enc_h2, enc_h2]

# Building a 2-layer LSTM decoder
dec_layer_1 = LSTM(latent_dim, return_sequences=True,
                   return_state=True, name="1st_layer_dec_LSTM")
dec_layer_2 = LSTM(latent_dim, return_sequences=True,
                   return_state=False, name="2nd_layer_dec_LSTM")
dec_dense = Dense(tgt_wordEmbed_dim, activation="softmax")
dec_inputs = Input(shape=(tgt_max_seq_length, tgt_wordEmbed_dim))
dec_outputs_1, dec_h1, dec_c1 = dec_layer_1(
    dec_inputs, initial_state=[enc_h1, enc_c1])
dec_outputs_2 = dec_layer_2(dec_outputs_1, initial_state = [enc_h2, enc_c2])
dec_outputs_final = dec_dense(dec_outputs_2)

#算出注意力權重
attention_scores = dot([dec_outputs_2, enc_outputs_2], axes = [2, 2])
attenton_weights = Activation("softmax")(attention_scores)
print("attention weights - shape: {}".format(attenton_weights.shape)) # shape: (None, enc_max_seq_length, dec_max_seq_length)

#利用加權平均求出context vector
context_vec = dot([attenton_weights, enc_outputs_2], axes = [2, 1])
print("context vector - shape: {}".format(context_vec.shape)) # shape: (None, dec_max_seq_length, latent_dim)

# 將解碼器當下的內部狀態與context vector連接起來，並得到注意力層的輸出
# concatenate context vector and decoder hidden state h_t
ht_context_vec = concatenate([context_vec, dec_outputs_2], name = "concatentated_vector")
print("ht_context_vec - shape: {}".format(ht_context_vec.shape)) # shape: (None, dec_max_seq_length, 2 * latent_dim)

# obtain attentional vector
attention_vec = Dense(latent_dim, use_bias = False, activation = "tanh", name = "attentional_vector")(ht_context_vec)
print("attention_vec - shape: {}".format(attention_vec.shape)) # shape: (None, dec_max_seq_length, latent_dim)

# 傳入softmax層預估當下輸出值的條件機率, 將注意力機制的輸出值傳入 softmax 層得到當下的目標詞向量
dec_outputs_final = Dense(tgt_wordEmbed_dim, use_bias = False, activation = "softmax")(attention_vec)
print("dec_outputs_final - shape: {}".format(dec_outputs_final.shape)) # shape: (None, dec_max_seq_length, tgt_wordEmbed_dim)


# Integrate seq2seq model with attention mechanism
seq2seq_2_layers_attention = Model([enc_inputs, dec_inputs], dec_outputs_final, name = "seq2seq_2_layers_attention")
seq2seq_2_layers_attention.summary()

# Preview model architecture
plot_model(seq2seq_2_layers_attention, to_file = "output/2-layer_seq2seq_attention.png", dpi = 100, show_shapes = True, show_layer_names = True)
