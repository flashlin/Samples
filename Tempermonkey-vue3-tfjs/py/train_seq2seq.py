import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, LSTM, Embedding
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
enc_layer_1 = LSTM(latent_dim, return_sequences = True, return_state = True, name = "1st_layer_enc_LSTM")
enc_layer_2 = LSTM(latent_dim, return_sequences = False, return_state = True, name = "2nd_layer_enc_LSTM")
enc_inputs = Input(shape = (src_max_seq_length, src_wordEmbed_dim))
enc_outputs_1, enc_h1, enc_c1 = enc_layer_1(enc_inputs)
enc_outputs_2, enc_h2, enc_c2 = enc_layer_2(enc_outputs_1)
enc_states = [enc_h1, enc_c1, enc_h2, enc_h2]

# Building a 2-layer LSTM decoder
dec_layer_1 = LSTM(latent_dim, return_sequences = True, return_state = True, name = "1st_layer_dec_LSTM")
dec_layer_2 = LSTM(latent_dim, return_sequences = True, return_state = True, name = "2nd_layer_dec_LSTM")
dec_dense = Dense(tgt_wordEmbed_dim, activation = "softmax")
dec_inputs = Input(shape = (tgt_max_seq_length, tgt_wordEmbed_dim))
dec_outputs_1, dec_h1, dec_c1 = dec_layer_1(dec_inputs, initial_state = [enc_h1, enc_c1])
dec_outputs_2, dec_h2, dec_c2 = dec_layer_2(dec_outputs_1, initial_state = [enc_h2, enc_c2])
dec_outputs_final = dec_dense(dec_outputs_2)

# Integrate seq2seq model
seq2seq_2_layers = Model([enc_inputs, dec_inputs], dec_outputs_2, name = "seq2seq_2_layers")
seq2seq_2_layers.summary()

plot_model(seq2seq_2_layers, to_file = "output/2-layer_seq2seq.png", dpi = 100, show_shapes = True, show_layer_names = True)
