import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Layer, Dense, LSTM, Embedding, Dot, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot
import pydot as pyd

# preparing hyperparameters

# source language- English
src_wordEmbed_dim = 18  # 詞向量維度18
src_max_seq_length = 100  # 句長最大值為 100

# target language- Spanish
tgt_wordEmbed_dim = 27  # dim of text vector representation
tgt_max_seq_length = 12  # max length of a sentence (including <SOS> and <EOS>)

# dim of context vector
latent_dim = 256  # LSTM 的內部狀態為 256維的向量



class LuongAttention(Layer):
    """
    Luong attention layer.
    """
    def __init__(self, latent_dim, tgt_wordEmbed_dim):
        super().__init__()
        self.AttentionFunction = Dot(axes = [2, 2], name = "attention_function")
        self.SoftMax = Activation("softmax", name = "softmax_attention")
        self.WeightedSum = Dot(axes = [2, 1], name = "weighted_sum")
        self.dense_tanh = Dense(latent_dim, use_bias = False, activation = "tanh", name = "dense_tanh")
        self.dense_softmax = Dense(tgt_wordEmbed_dim, use_bias = False, activation = "softmax", name = "dense_softmax")

    def call(self, enc_outputs_top, dec_outputs_top):
        attention_scores = self.AttentionFunction([enc_outputs_top, dec_outputs_top])
        attenton_weights = self.SoftMax(attention_scores)
        context_vec = self.WeightedSum([attenton_weights, enc_outputs_top])
        ht_context_vec = concatenate([context_vec, dec_outputs_top], name = "concatentated_vector")
        attention_vec = self.dense_tanh(ht_context_vec)
        return attention_vec


class Encoder(Layer):
    """
    2-layer Encoder LSTM with/ without attention mechanism.
    """
    def __init__(self, latent_dim, src_wordEmbed_dim, src_max_seq_length):
        super().__init__()
        # self.inputs = Input(shape = (src_max_seq_length, src_wordEmbed_dim), name = "encoder_inputs")
        self.latent_dim = latent_dim
        self.embedding_dim = src_wordEmbed_dim
        self.max_seq_length = src_max_seq_length
        self.lstm_input = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "1st_layer_enc_LSTM")
        self.lstm = LSTM(units = latent_dim, return_sequences = False, return_state = True, name = "2nd_layer_enc_LSTM")
        self.lstm_return_seqs = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "2nd_layer_enc_LSTM")

    def call(self, inputs, withAttention = False):
        outputs_1, h1, c1 = self.lstm_input(inputs)
        if withAttention:
            outputs_2, h2, c2 = self.lstm_return_seqs(outputs_1)
        else:
            outputs_2, h2, c2 = self.lstm(outputs_1)
        states = [h1, c1, h2, h2]
        return outputs_2, states


class Decoder(Layer):
    """
    2-layer Decoder LSTM with/ without attention mechanism.
    """
    def __init__(self, latent_dim, tgt_wordEmbed_dim, tgt_max_seq_length):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = tgt_wordEmbed_dim
        self.max_seq_length = tgt_max_seq_length
        self.lstm_input = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "1st_layer_dec_LSTM")
        self.lstm_return_no_states = LSTM(units = latent_dim, return_sequences = True, return_state = False, name = "2nd_layer_dec_LSTM")
        self.lstm = LSTM(units = latent_dim, return_sequences = True, return_state = True, name = "2nd_layer_dec_LSTM")
        self.dense = Dense(tgt_wordEmbed_dim, activation = "softmax", name = "softmax_dec_LSTM")

    def call(self, inputs, enc_states, enc_outputs_top = None, withAttention = False):
        # unpack encoder states [h1, c1, h2, c2]
        enc_h1, enc_c1, enc_h2, enc_c2 = enc_states

        outputs_1, h1, c1 = self.lstm_input(inputs, initial_state = [enc_h1, enc_c1])
        if withAttention:
            # instantiate Luong attention layer
            attention_layer = LuongAttention(latent_dim = self.latent_dim, tgt_wordEmbed_dim = self.max_seq_length)

            dec_outputs_top = self.lstm_return_no_states(outputs_1, initial_state = [enc_h2, enc_c2])
            attention_vec = attention_layer(dec_outputs_top, enc_outputs_top)
            outputs_final = self.dense_softmax(attention_vec)
        else:
            outputs_2, h2, c2 = self.lstm(outputs_1, initial_state = [enc_h2, enc_c2])
            outputs_final = self.dense(outputs_2)
        return outputs_final


class My_Seq2Seq(Model):
    """
    2-Layer LSTM Encoder-Decoder with/ without Luong attention mechanism.
    """
    def __init__(self, latent_dim, src_wordEmbed_dim, src_max_seq_length, tgt_wordEmbed_dim, tgt_max_seq_length, model_name = None, withAttention = False,
 input_text_processor = None, output_text_processor = None):
        super().__init__(name = model_name)
        self.encoder = Encoder(latent_dim, src_wordEmbed_dim, src_max_seq_length)
        self.decoder = Decoder(latent_dim, tgt_wordEmbed_dim, tgt_max_seq_length)
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor
        self.withAttention = withAttention


    def call(self, enc_inputs, dec_inputs):
        enc_outputs, enc_states = self.encoder(enc_inputs)
        dec_outputs = self.decoder(inputs = dec_inputs, enc_states = enc_states, enc_outputs_top = enc_outputs, withAttention = self.withAttention)
        return dec_outputs

    def plot_model_arch(self, enc_inputs, dec_inputs, outfile_path = None):
        tmp_model = Model(inputs = [enc_inputs, dec_inputs], outputs = self.call(enc_inputs, dec_inputs))
        plot_model(tmp_model, to_file = outfile_path, dpi = 100, show_shapes = True, show_layer_names = True)


if __name__ == "__main__":

    # hyperparameters
    src_wordEmbed_dim = 18
    src_max_seq_length = 4
    tgt_wordEmbed_dim = 27
    tgt_max_seq_length = 12
    latent_dim = 256

    # specifying data shapes
    enc_inputs = Input(shape = (src_max_seq_length, src_wordEmbed_dim))
    dec_inputs = Input(shape = (tgt_max_seq_length, tgt_wordEmbed_dim))

    # instantiate My_Seq2Seq class
    seq2seq = My_Seq2Seq(latent_dim, src_wordEmbed_dim, src_max_seq_length, tgt_wordEmbed_dim, tgt_max_seq_length, withAttention = True, model_name = "seq2seq_no_attention")
    # build model
    dec_outputs = seq2seq(
        enc_inputs = Input(shape = (src_max_seq_length, src_wordEmbed_dim)),
        dec_inputs = Input(shape = (tgt_max_seq_length, tgt_wordEmbed_dim))
        )
    print("model name: {}".format(seq2seq.name))
    
    seq2seq.summary()
    seq2seq.plot_model_arch(enc_inputs, dec_inputs, outfile_path = "output/seq2seq_LSTM_with_attention.png")


