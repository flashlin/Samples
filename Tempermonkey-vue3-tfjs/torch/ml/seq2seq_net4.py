import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

from common.io import info
from ml.lit import PositionalEncoding, BaseLightning


class TransformerModel(nn.Module):
    def __init__(self,
                 n_token: int,
                 d_model: int,
                 n_head: int,
                 d_hid: int,
                 n_layers: int,
                 dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_token)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.words.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """生成一个负无穷大的上三角矩阵，对角线上元素为 0。"""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


# 很難訓練, loss=240
class LitTransformer(BaseLightning):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        hidden_size = 1024
        # self.model = TransformerModel(n_token=vocab.get_size(),
        #                               d_model=1024,
        #                               n_head=8,
        #                               d_hid=1024,
        #                               n_layers=6)
        self.model = TransformerModel(n_token=vocab.get_size(),
                                      d_model=4096,
                                      n_head=8,
                                      d_hid=1024,
                                      n_layers=2)
        self.fc = nn.Linear(hidden_size * 2, vocab.get_size())
        self.criterion = nn.CrossEntropyLoss()  # reduction="none")

    def forward(self, batch):
        device = self.get_device()
        src, src_len, tgt, tgt_len = batch
        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)
        max_seq_length = src.size(0)
        src_mask = generate_square_subsequent_mask(max_seq_length).to(device)
        # info(f" {src.shape=} {src_mask.shape=}")
        logits = self.model(src, src_mask)
        return logits, tgt

    def _calculate_loss(self, batch, batch_idx):
        (logits, y) = batch
        logits = logits.view(-1, self.vocab.get_size())
        y = torch.flatten(y)
        # y = y.squeeze(1)
        loss = self.criterion(logits, y)
        return loss

    def infer(self, text):
        device = next(self.parameters()).device
        text_values = self.vocab.encode_to_tokens(text)
        seq_length = len(text_values)
        src_mask = generate_square_subsequent_mask(seq_length).to(device)
        text_values = torch.tensor([text_values]).to(device).permute(1, 0)
        with torch.no_grad():
            output = self.model(text_values, src_mask)
            output_flat = output.view(-1, self.vocab.get_size())
        output_flat = output_flat.argmax(1).tolist()
        return self.vocab.decode(output_flat)
