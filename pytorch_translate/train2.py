import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from data_utils import load_csv_to_dataframe, split_dataframe

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)

        # GRU layer. The input and output are both of the same size
        #  since embedding size = hidden size in this example
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # The inputs are first transformed into embeddings
        embedded = self.embedding(input)
        output = embedded

        # As in any RNN, the new input and the previous hidden states are fed
        #  into the model at each time step
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        # This method is used to create the innitial hidden states for the encoder
        return torch.zeros(1, batch_size, self.hidden_size)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)

        # The GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size)

        # Fully-connected layer for scores
        self.out = nn.Linear(hidden_size, output_size)

        # Applying Softmax to the scores
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Feeding input through embedding layer
        output = self.embedding(input)

        # Applying an activation function (ReLu)
        output = F.relu(output)

        # Feeding input and previous hidden state
        output, hidden = self.gru(output, hidden)

        # Outputting scores from the final time-step
        output = self.softmax(self.out(output[0]))

        return output, hidden

    # We do not need an .initHidden() method for the decoder since the
    #  encoder output will act as input in the first decoder time-step


if __name__ == '__main__':
    data_df = load_csv_to_dataframe()
    train_df, val_df = split_dataframe(data_df)

    src_raw_train = data_df["source_sentence"]
    tgt_raw_train = data_df["target_sentence"]

    src_train = [sent.strip().split(" ") for sent in src_raw_train]
    tgt_train = [sent.strip().split(" ") for sent in tgt_raw_train]

    src_index2word = ["<PAD>", "<SOS>", "<EOS>"]
    tgt_index2word = ["<PAD>", "<SOS>", "<EOS>"]

    for ds in [src_train, tgt_train]:
        for sent in ds:
            for token in sent:
                if token not in src_index2word:
                    src_index2word.append(token)

    src_word2index = {token: idx for idx, token in enumerate(src_index2word)}
    tgt_index2word = src_index2word
    tgt_word2index = {token: idx for idx, token in enumerate(tgt_index2word)}

    print(f"{src_index2word=}")

    src_lengths = sum([len(sent) for sent in src_train]) / len(src_train)
    seq_length = 100

    def encode_and_pad(vocab, sent, max_length):
        sos = [vocab["<SOS>"]]
        eos = [vocab["<EOS>"]]
        pad = [vocab["<PAD>"]]
        if len(sent) < max_length - 2:  # -2 for SOS and EOS
            n_pads = max_length - 2 - len(sent)
            encoded = [vocab[w] for w in sent]
            return sos + encoded + eos + pad * n_pads
        else:  # sent is longer than max_length; truncating
            encoded = [vocab[w] for w in sent]
            truncated = encoded[:max_length - 2]
            return sos + truncated + eos


    src_train_encoded = [encode_and_pad(src_word2index, sent, seq_length) for sent in src_train]
    tgt_train_encoded = [encode_and_pad(tgt_word2index, sent, seq_length) for sent in tgt_train]

    batch_size = 2
    train_x = np.array(src_train_encoded)
    train_y = np.array(tgt_train_encoded)
    #test_x = np.array(en_test_encoded)
    #test_y = np.array(de_test_encoded)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    #test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    #test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)

    hidden_size = 128
    encoder = EncoderRNN(len(src_index2word), hidden_size).to(DEVICE)
    decoder = DecoderRNN(hidden_size, len(tgt_index2word)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=3e-3)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-3)

    ####
    print(f"start train")
    input_length = target_length = seq_length

    SOS = src_word2index["<SOS>"]
    EOS = src_word2index["<EOS>"]
    epochs = 15

    device = DEVICE
    losses = []
    for epoch in range(epochs):
        for idx, batch in enumerate(train_dl):

            # Creating initial hidden states for the encoder
            encoder_hidden = encoder.initHidden()

            # Sending to device
            encoder_hidden = encoder_hidden.to(device)

            # Assigning the input and sending to device
            input_tensor = batch[0].to(device)

            # Assigning the output and sending to device
            target_tensor = batch[1].to(device)

            # Clearing gradients
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()

            # Enabling gradient calculation
            with torch.set_grad_enabled(True):

                # Feeding batch into encoder
                encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

                # This is a placeholder tensor for decoder outputs. We send it to device as well
                dec_result = torch.zeros(target_length, batch_size, len(tgt_index2word)).to(device)

                # Creating a batch of SOS tokens which will all be fed to the decoder
                decoder_input = target_tensor[:, 0].unsqueeze(dim=0).to(device)

                # Creating initial hidden states of the decoder by copying encoder hidden states
                decoder_hidden = encoder_hidden

                # For each time-step in decoding:
                for i in range(1, target_length):
                    # Feed input and previous hidden states
                    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

                    # Finding the best scoring word
                    best = decoder_output.argmax(1)

                    # Assigning next input as current best word
                    decoder_input = best.unsqueeze(dim=0)

                    # Creating an entry in the placeholder output tensor
                    dec_result[i] = decoder_output

                # Creating scores and targets for loss calculation
                print(f"{dec_result.shape=}")
                scores = dec_result.transpose(1, 0)[1:].reshape(-1, dec_result.shape[2])
                targets = target_tensor[1:].reshape(-1)

                # Calculating loss
                loss = criterion(scores, targets)

                # Performing backprop and clipping excess gradients
                loss.backward()

                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

                enc_optimizer.step()
                dec_optimizer.step()

                # Keeping track of loss
                losses.append(loss.item())
                if idx % 100 == 0:
                    print(idx, sum(losses) / len(losses))

    plt.plot(losses)

    ####
    test_sentence = "from tb1 in user select tb1"
    print(f"{test_sentence=}")
    # Tokenizing, Encoding, transforming to Tensor
    test_sentence = torch.tensor(encode_and_pad(src_word2index, test_sentence.split(), seq_length)).unsqueeze(dim=0)

    encoder_hidden = torch.zeros(1, 1, hidden_size)
    encoder_hidden = encoder_hidden.to(device)

    input_tensor = test_sentence.to(device)

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    result = []

    encoder_outputs = torch.zeros(seq_length, encoder.hidden_size, device=device)
    with torch.set_grad_enabled(False):
        encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

        dec_result = torch.zeros(target_length, 1, len(tgt_index2word)).to(device)

        decoder_input = torch.tensor([SOS]).unsqueeze(dim=0).to(device)
        decoder_hidden = encoder_hidden
        for di in range(1, target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            best = decoder_output.argmax(1)
            result.append(tgt_index2word[best.to('cpu').item()])
            if best.item() == EOS:
                break

            decoder_input = best.unsqueeze(dim=0)
            dec_result[di] = decoder_output

        scores = dec_result.reshape(-1, dec_result.shape[2])
        #targets = target_tensor.reshape(-1)

    answer = " ".join(result)
    print(f"{answer=}")
