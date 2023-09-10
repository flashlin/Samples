import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from common.io import info
from ml.data_utils import pad_list
from ml.lit import BaseLightning


class Seq2Seq(nn.Module):
    def __init__(self, vocab, encoder, decoder, max_length=500):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        device = next(self.parameters()).device
        input_length = source.size(1)  # get the input length (number of words in sentence)
        batch_size = target.shape[0]
        target_length = target.shape[1]
        vocab_size = self.decoder.output_dim

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(batch_size, target_length, vocab_size).to(device)
        # outputs = outputs.permute(1, 0, 2)

        # use the encoder’s hidden layer as the decoder hidden
        # decoder_hidden = encoder_hidden.to(device)

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
            # info(f" {t=} {decoder_input.shape=}")
            decoder_output, decoder_hidden = self.decoder(decoder_input, encoder_output, decoder_hidden)
            # info(f" {t=} {decoder_output.shape=}")
            # decoder_output = decoder_output.view(batch_size, 300, -1)
            # outputs[:, t, :] = decoder_output[:, t, :] #.clone().detach()  unsqueeze(-1)
            outputs[:, t] = decoder_output
            top = decoder_output.argmax(1)
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

        hidden_state = torch.randn(1 * 2, batch_size,
                                   self.n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size,
                                 self.n_hidden)  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)
        return model


def create_src_lengths_mask(batch_size, src_lengths, max_src_len=None):
    """
    :param batch_size:
    :param src_lengths:
    :param max_src_len:
    :return: [batch_size, max_seq_len]
    """
    if max_src_len is None:
        max_src_len = int(src_lengths.max())
    src_indices = torch.arange(0, max_src_len).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_src_len)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_src_len)
    return (src_indices < src_lengths).int().detach()


def masked_softmax(scores, src_lengths, src_length_masking=True):
    if src_length_masking:
        bsz, max_src_len = scores.size()
        src_mask = create_src_lengths_mask(bsz, src_lengths)
        scores = scores.masked_fill(src_mask == 0, -np.inf)
    return F.softmax(scores, dim=-1).type_as(scores)


class MLPAttentionNetwork(nn.Module):
    def __init__(self, hidden_dim, attention_dim, src_length_masking=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.src_length_masking = src_length_masking
        self.proj_w = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.proj_v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, x, x_lengths):
        batch_size, seq_len, _ = x.size()
        flat_inputs = x.reshape(-1, self.hidden_dim)
        mlp_x = self.proj_w(flat_inputs)
        att_scores = self.proj_v(mlp_x).view(batch_size, seq_len)
        normalized_masked_att_scores = masked_softmax(att_scores, x_lengths, self.src_length_masking)
        attn_x = (x * normalized_masked_att_scores.unsqueeze(2)).sum(0)
        return normalized_masked_att_scores, attn_x


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, embbed_dim)
        # self.rnn = nn.GRU(embbed_dim, hidden_dim, num_layers=num_layers)
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
        # self.rnn = nn.GRU(hidden_dim * 2 + self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.rnn = nn.LSTM(hidden_dim * 2 + embbed_dim, hidden_dim, num_layers, batch_first=True)
        self.energy = nn.Linear(hidden_dim * 3, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.LogSoftmax(dim=1)
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


class Seq2SeqNet(BaseLightning):
    def __init__(self, vocab, embbed_dim=256, hidden_dim=512):
        super().__init__()
        self.vocab = vocab
        n_layers = 1
        encoder = Encoder(input_dim=vocab.get_size(), hidden_dim=hidden_dim, embbed_dim=embbed_dim, num_layers=n_layers)
        decoder = Decoder(output_dim=vocab.get_size(), hidden_dim=hidden_dim, embbed_dim=embbed_dim,
                          num_layers=n_layers)
        self.model = Seq2Seq(vocab, encoder=encoder, decoder=decoder)
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
        text_to_indices = self.vocab.encode_to_tokens(text)
        return self.model.infer(text_to_indices, len(text))
