import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import nltk

from common.io import info
from ml.lit import BaseLightning

NEG_INF = -10000
TINY_FLOAT = 1e-6


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF, dim=1)

    return mask_max


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask


class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]
    seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort).to('cpu')

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort, batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        return y


class QuoraModel(nn.Module):
    """Model for quora insincere question classification.
    """

    def __init__(self, args):
        super(QuoraModel, self).__init__()

        vocab_size = args["vocab_size"]
        pretrained_embed = args["pretrained_embed"]
        padding_idx = args["padding_idx"]
        embed_dim = 300
        num_classes = 1
        num_layers = 2
        hidden_dim = 50
        dropout = 0.5

        if pretrained_embed is None:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = nn.Embedding.from_pretrained(
                pretrained_embed, freeze=False)
        self.embed.padding_idx = padding_idx

        self.rnn = DynamicLSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=True)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 6, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, num_classes)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, word_seq, seq_len):
        # mask
        max_seq_len = torch.max(seq_len)
        mask = seq_mask(seq_len, max_seq_len)  # [b,msl]

        # embed
        e = self.drop(self.embed(word_seq))  # [b,msl]->[b,msl,e]

        # bi-rnn
        r = self.rnn(e, seq_len)  # [b,msl,e]->[b,msl,h*2]

        # attention
        att = self.fc_att(r).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        r_att = torch.sum(att.unsqueeze(-1) * r, dim=1)  # [b,h*2]

        # pooling
        r_avg = mask_mean(r, mask)  # [b,h*2]
        r_max = mask_max(r, mask)  # [b,h*2]
        r = torch.cat([r_avg, r_max, r_att], dim=-1)  # [b,h*6]

        # feed-forward
        f = self.drop(self.act(self.fc(r)))  # [b,h*6]->[b,h]
        logits = self.out(f).squeeze(-1)  # [b,h]->[b]

        return logits


class SeqToOneClassificationLstm(BaseLightning):
    def __init__(self, vocab_size, padding_idx=0, pretrained_embed=None):
        super(SeqToOneClassificationLstm, self).__init__()
        model_args = {
            "vocab_size": vocab_size,
            "padding_idx": padding_idx,
            "pretrained_embed": pretrained_embed,
        }
        self.model = QuoraModel(model_args)

    def forward(self, batch):
        device = self.get_device()
        src, src_len, tgt, tgt_len = batch
        src_len = torch.tensor(src_len, dtype=torch.long).to(device)
        logits = self.model(src, src_len)
        # loss = network.loss(logits, target.float())
        return logits, tgt

    def _calculate_loss(self, batch, batch_idx, mode="train"):
        logits, tgt = batch
        info(f" {logits.shape=} {tgt.shape=}")
        loss = self.model.loss(logits, tgt)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text, threshold=0.33):
        batch_output = []
        with torch.no_grad():
            prediction = self.model(text)
        batch_output.append(prediction.detach())
        return self._post_processor(batch_output, threshold)

    def _post_processor(self, batch_output, threshold=0.33):
        y_preds = []
        for logit in batch_output:
            y_pred = (torch.sigmoid(logit) > threshold).long().cpu().numpy()
            y_preds.append(y_pred)
        y_pred = np.concatenate(y_preds, axis=0)
        return y_pred


