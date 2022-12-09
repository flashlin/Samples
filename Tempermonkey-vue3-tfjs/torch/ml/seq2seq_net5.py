import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from common.io import info
from ml.data_utils import pad_list
from ml.lit import BaseLightning


class Seq2Seq(nn.Module):
    def __init__(self, vocab, encoder, decoder, max_length):
        super().__init__()
        self.max_length = max_length
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab
        self.prev_fc = nn.Linear(self.encoder.embbed_dim, vocab.get_size())

    def embedding(self, x):
        return self.encoder.embedding(x).view(-1, self.encoder.embbed_dim)

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        device = next(self.parameters()).device
        input_length = source.size(1)  # get the input length (number of words in sentence)
        batch_size = target.shape[0]
        target_length = target.shape[1]
        vocab_size = self.decoder.output_dim

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(batch_size, target_length, vocab_size).to(device)
        # outputs = outputs.permute(1, 0, 2)

        src = self.embedding(source)

        # encode every word in a sentence
        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(source[:, i])
        # add a token before the first predicted word
        # start_input = pad_list([self.vocab.bos_idx], 300)
        # start_input = [self.vocab.bos_idx]
        # decoder_input = torch.tensor([start_input] * batch_size, dtype=torch.long, device=device)
        decoder_input = target[:, 0]
        decoder_hidden = encoder_hidden

        # torch.autograd.set_detect_anomaly(True)
        for t in range(1, target_length):
            # info(f" {t=} {decoder_input.shape=} {decoder_input.dtype=} {encoder_output.shape=} {encoder_output.dtype=} {decoder_hidden[0].shape=}")
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_output, decoder_hidden)
            # info(f" {t=} {decoder_output.shape=}")
            # decoder_output = decoder_output.view(batch_size, 300, -1)
            # outputs[:, t, :] = decoder_output[:, t, :] #.clone().detach()  unsqueeze(-1)
            outputs[:, t] = decoder_output
            top = decoder_output.argmax(1)

            # top = self.embedding(top.unsqueeze(0))
            # prev_cells = torch.cat((src, top))
            # top = self.prev_fc(prev_cells).argmax(1)[-1]
            # top = torch.tensor([top]).to(device)

            teacher_force = random.random() < teacher_forcing_ratio
            decoder_input = target[:, t] if teacher_force else top
            # top_value, top_index = decoder_output.max(dim=-1)
            # info(f" {target[:, t].requires_grad=}")
            # decoder_input[:, t] = t_tgt if teacher_force else top_index[:, t]

        # outputs = outputs.argmax(dim=-1, keepdim=False)
        # info(f" {outputs.shape=} {outputs.requires_grad=}")
        # outputs = outputs #.clone().detach()
        return outputs

    def infer(self, text_to_indices, max_length):
        device = next(self.parameters()).device
        input_length = len(text_to_indices)
        sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(0).to(device)

        with torch.no_grad():
            for t in range(input_length):
                encoder_output, hidden_states = self.encoder(sentence_tensor[0, t])

        vocab = self.vocab
        outputs = [vocab.get_value('<bos>')]
        for _ in range(max_length):
            previous_word = torch.LongTensor([outputs[-1]]).to(device)

            with torch.no_grad():
                output, hidden_states = self.decoder(
                    previous_word, encoder_output, hidden_states
                )
                best_guess = output.argmax(1).item()

            outputs.append(best_guess)

            # Model predicts it's the end of the sentence
            if output.argmax(1).item() == vocab.get_value('<eos>'):
                break

        translated_sentence = vocab.decode(outputs)
        return translated_sentence


class BiLSTM(nn.Module):
    def __init__(self, n_class, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, x):
        """
        :param x: [batch_size, max_len, n_class]
        :return: [batch_size, n_class]
        """
        batch_size = x.shape[0]
        input = x.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1 * 2, batch_size, self.n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size, self.n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)
        return model


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embbed_dim)
        self.rnn = nn.LSTM(embbed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src).view(1, -1, self.embbed_dim)
        outputs, (h0, c0) = self.rnn(embedded)

        h0 = self.fc_hidden(torch.cat((h0[0:1], h0[1:2]), dim=2))
        c0 = self.fc_cell(torch.cat((c0[0:1], c0[1:2]), dim=2))

        return outputs, (h0, c0)


class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super().__init__()
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.rnn = nn.LSTM(hidden_dim * 2 + embbed_dim, hidden_dim, num_layers, batch_first=True)
        self.energy = nn.Linear(hidden_dim * 3, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_output, hidden_states):
        x = x.unsqueeze(0)
        hidden, cell = hidden_states

        embedded = self.embedding(x)
        sequence_length = encoder_output.shape[1]

        # assert sequence_length == 1, f"{sequence_length=} is not correct."
        assert hidden.shape == (1, 1, 512), f"{hidden.shape=} is not correct."
        # h_reshaped = hidden.repeat(sequence_length, 1, 1) #.permute(1, 0, 2)
        h_reshaped = hidden.repeat(1, sequence_length, 1)

        # info(f" cat {h_reshaped.shape=} {encoder_output.shape=}")
        energy = torch.cat((h_reshaped, encoder_output), dim=2)
        energy = self.relu(self.energy(energy))

        attention = self.softmax(energy)
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_output)

        # info(f" {context_vector.shape=} {embedded.shape=}")
        # assert context_vector.shape == (1, 1, 1024), f"{context_vector.shape=} is not correct."
        rnn_input = torch.cat((context_vector, embedded), dim=2)

        outputs, (hidden_states, cell) = self.rnn(rnn_input, hidden_states)
        predictions = self.fc(outputs).squeeze(0)

        # xnput = encoder_output.view(1, -1)
        # output, hidden = self.gru(embedded, hidden)
        # predictions = self.softmax(self.out(output[0]))
        return predictions, (hidden, cell)


class MySeq2SeqNet(BaseLightning):
    def __init__(self, vocab, embbed_dim=256, hidden_dim=512, n_layers=1):
        super().__init__()
        self.vocab = vocab
        encoder = Encoder(input_dim=vocab.get_size(), hidden_dim=hidden_dim, embbed_dim=embbed_dim, num_layers=n_layers)
        decoder = Decoder(output_dim=vocab.get_size(), hidden_dim=hidden_dim, embbed_dim=embbed_dim, num_layers=n_layers)
        self.model = Seq2Seq(vocab, encoder=encoder, decoder=decoder, max_length=500)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx)

    def forward(self, batch):
        src, src_lens, tgt, tgt_lens = batch
        return self.model(src, tgt), tgt

    def _calculate_loss(self, batch, batch_idx):
        x_hat, y = batch

        x_hat = x_hat.view(-1, x_hat.shape[-1])
        y = y.view(-1)

        return self.loss_fn(x_hat, y)

    def infer(self, text):
        text_to_indices = self.vocab.encode(text)
        return self.model.infer(text_to_indices, len(text))
