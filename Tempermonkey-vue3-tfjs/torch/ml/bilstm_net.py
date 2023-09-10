import torch
from torch import nn as nn
from torch.autograd import Variable


class EncoderBiLSTM(nn.Module):
    def __init__(self, hidden_size, embedding_layer, n_layers=2, level='local'):
        super().__init__()
        self.level = level
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if self.level == 'local':
            self.embedding = embedding_layer
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, inputs, hidden):
        device = next(self.parameters()).device
        # embedded is of size (n_batch, seq_len, emb_dim)
        # lstm needs: (seq_len, batch, input_size)
        # lstm output: (seq_len, batch, hidden_size * num_directions)
        if self.level == 'local':
            embedded = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            # local encoder: input is (rt, re, rm)
            inp = embedded
            # inp (seq_len, batch, emb_dim)
            seq_len = embedded.size(1)
            batch_size = embedded.size(0)
            embed_dim = embedded.size(2)
            outputs = Variable(torch.zeros(seq_len, batch_size, embed_dim))
            outputs = outputs.to(device)

            for ei in range(seq_len):
                if ei > 0 and ei % 32 == 0:
                    hidden = self.initHidden(batch_size)
                seq_i = inp[:, ei, :].unsqueeze(0)
                # inputs of size: (1, batch, emb_dim)
                output, hidden = self.bilstm(seq_i, hidden)
                # output of size: (1, batch, emb_dim)
                outputs[:, ei, :] = output[:, 0, :]
        else:
            inp = inputs['local_hidden_states']
            outputs, hidden = self.bilstm(inp, hidden)
        return outputs, hidden


class EncoderBiLSTMMaxPool(nn.Module):
    def __init__(self, hidden_size, embedding_layer, n_layers=2, level='local'):
        super().__init__()
        self.level = level
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        if self.level == 'local':
            self.embedding = embedding_layer
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=n_layers, batch_first=True, bidirectional=True)

    def forward(self, inputs, hidden):
        """
        :param inputs: (batch, seq_len, input_dim)
        :param hidden:
        :return:
        """
        device = next(self.parameters()).device
        if self.level == 'local':
            # embedded is of size (n_batch, seq_len, emb_dim)
            embedded = self.embedding(inputs['rt'], inputs['re'], inputs['rm'])
            inp = embedded
            seq_len = embedded.size(1)
            batch_size = embedded.size(0)
            embed_dim = embedded.size(2)
            bilstm_outs = Variable(torch.zeros(seq_len, batch_size, embed_dim))
            bilstm_outs = bilstm_outs.to(device)

            for ei in range(seq_len):
                if ei > 0 and ei % 32 == 0:
                    hidden = self.initHidden(batch_size)

                inputs = inp[ei, :, :].unsqueeze(0)
                # inputs of size: (1, batch, emb_dim)
                outputs, hidden = self.bilstm(inputs, hidden)
                # output of size: (1, batch, emb_dim)
                bilstm_outs[ei, :, :] = outputs[0, :, :]

        else:
            inp = inputs['local_hidden_states']
            bilstm_outs, nh = self.bilstm(inp, hidden)

        # bilstm_outs: (seq_len, batch, hidden_size * num_directions )
        output = bilstm_outs.permute(1, 2, 0)
        # bilstm_outs: (batch, hidden_size * num_directions, seq_len)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        return bilstm_outs, output

    def initHidden(self, batch_size):
        device = next(self.parameters()).device
        forward = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size // 2), requires_grad=False)
        backward = Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size // 2), requires_grad=False)
        return forward.to(device), backward.to(device)


class HierarchicalBiLSTM(nn.Module):
    def __init__(self, hidden_size, local_embed, n_layers=2):
        super(HierarchicalBiLSTM, self).__init__()
        self.LocalEncoder = EncoderBiLSTMMaxPool(hidden_size, local_embed, n_layers=n_layers, level='local')
        self.GlobalEncoder = EncoderBiLSTMMaxPool(hidden_size, None, n_layers=n_layers, level='global')


