from typing import Tuple

import torch
from torch import nn, Tensor
from common.io import info
from ml.lit import BaseLightning, start_train, PositionalEncoding
from prepare6 import Linq2TSqlDataset
from utils.tokenizr import VOCAB_SIZE
import math


def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # return mask

class TransformerModel(nn.Module):
    def __init__(self, n_token, d_model, n_head, d_hid, n_layers, dropout=0.2):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(n_token, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, n_token)
        self.init_weights()

    def init_weights(self):
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

def get_bptt_batch(source: Tensor, i: int, bptt: int=35) -> Tuple[Tensor, Tensor]:
    """ 將數據切成 bptt 長度的小段
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


class LitSeq2Seq(BaseLightning):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=512, hidden_dim=3, dropout=0.2):
        super().__init__()
        self.n_tokens = vocab_size
        self.model = TransformerModel(self.n_tokens, d_model=200, n_head=2, d_hid=200, n_layers=2, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()
        ds = Linq2TSqlDataset('../output/linq-sample.csv')
        train_loader, val_loader = ds.create_dataloader()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = val_loader
        self.criterion = nn.NLLLoss()
        self.src_mask = generate_square_subsequent_mask(35)

    def training_step(self, batch, batch_idx):
        """
        Back Propagation Through Time (BPTT)
        """
        bptt = 35
        src_mask = self.src_mask
        total_loss = 0.
        # num_batches = len(batch) // bptt
        for bptt_batch, i in enumerate(range(0, batch.size(0) - 1, bptt)):
            data, targets = get_bptt_batch(batch, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = self.model(data, src_mask)
            loss = self._calculate_loss(output.view(-1, self.n_tokens), mode="train")
            total_loss += loss
        return total_loss

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def _calculate_loss(self, batch, mode="train"):
        y_hat, y = batch
        info(f" {self.n_tokens=} {y_hat.shape=} {y.shape=}")
        info(f" {y_hat.view(-1, self.n_tokens).shape=} {y.shape=}")
        self.criterion(y_hat.view(-1, self.n_tokens), y)
        loss = self.loss_compute(y_hat, y, self.tgt_vocab_size)
        self.log("%s_loss" % mode, loss)
        # y_hat, y = batch
        # acc = self._compute_accuracy(y_hat, y)
        # self.log("%s_acc" % mode, acc)
        return loss

    # def _compute_accuracy(self, y_hat, y):
    #     return accuracy(y_hat, y)

    def infer(self):
        # pred = model(seq_in)
        pred = 1.0
        # pred = to_prob(F.softmax(pred).data[0].numpy())  # softmax 後轉成機率分佈
        # char = np.random.choice(chars, p=pred)  # 依機率分佈選字
        pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # start_train(LitSeq2Seq, device='cpu', src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)
    start_train(LitSeq2Seq, device='cpu')

if __name__ == "__main__":
    main()