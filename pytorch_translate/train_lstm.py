import time
import numpy as np
import torch
# from keras.models import Model
# from keras.layers import Input, LSTM, Dense
# from keras.optimizers import Adam
# from keras.utils import *
# from keras.initializers import *
import tensorflow as tf

from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from data_utils import load_csv_to_dataframe, split_dataframe

# import warnings
# warnings.filterwarnings("ignore")

# data manupulation libs

from train_utils import build_vocab_from_dataframe, generate_batch, train_epoch, evaluate

# from pandarallel import pandarallel
# Initialization
# pandarallel.initialize()

#Hyperparameters
batch_size = 64
latent_dim = 256
num_samples = 10000


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    data_df = load_csv_to_dataframe()
    train_df, val_df = split_dataframe(data_df)

    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_chars = set()
    target_chars = set()

    for src_sentence, tgt_sentence in zip(train_df["source_sentence"], train_df["target_sentence"]):
        tgt_sentence = "\t" + tgt_sentence + "\t"
        input_texts.append(src_sentence)
        target_texts.append(tgt_sentence)
        for char in src_sentence:
            if char not in input_chars:
                input_chars.add(char)
        for char in tgt_sentence:
            if char not in target_chars:
                target_chars.add(char)

    input_chars = sorted(list(input_chars))
    target_chars = sorted(list(target_chars))
    num_encoder_tokens = len(input_chars)
    num_decoder_tokens = len(target_chars)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print(f"{input_chars=}")
    print(f"{num_encoder_tokens=} {max_encoder_seq_length=}")
    print(f"{num_decoder_tokens=} {max_decoder_seq_length=}")

    # Define data for encoder and decoder
    input_token_id = dict([(char, i) for i, char in enumerate(input_chars)])
    target_token_id = dict([(char, i) for i, char in enumerate(target_chars)])

    encoder_in_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    decoder_in_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

    decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_in_data[i, t, input_token_id[char]] = 1.
        for t, char in enumerate(target_text):
            decoder_in_data[i, t, target_token_id[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_id[char]] = 1.

    # Define and process the input sequence
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Using `encoder_states` set up the decoder as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Final model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Model Summary
    model.summary()

    # Model data Shape
    print("encoder_in_data shape:", encoder_in_data.shape)
    print("decoder_in_data shape:", decoder_in_data.shape)
    print("decoder_target_data shape:", decoder_target_data.shape)

    # Compiling and training the model
    model.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.001), loss='categorical_crossentropy')

    model.fit([encoder_in_data, decoder_in_data], decoder_target_data, batch_size=batch_size, epochs=20,
              validation_split=0.2)

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    reverse_input_char_index = dict((i, char) for char, i in input_token_id.items())
    reverse_target_char_index = dict((i, char) for char, i in target_token_id.items())

    # Define Decode Sequence
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Get the first character of target sequence with the start character.
        target_seq[0, 0, target_token_id['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]
        return decoded_sentence


    for seq_index in range(10):
        input_seq = encoder_in_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

