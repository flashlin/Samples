import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from common.io import info
from preprocess_data import Seq2SeqDataset
from lit import BaseLightning, start_train, PositionalEncoding, CosineWarmupScheduler, MultiHeadAttention
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE

class SeqEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, pe_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional = PositionalEncoding(pe_dim)

    def forward(self, x):
        output = self.embedding(x)
        output = self.positional(output)
        return output

class TranslateModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embedding = SeqEmbedding(src_vocab_size)
        self.tgt_embedding = SeqEmbedding(tgt_vocab_size)
        self.transformer = nn.Transformer(d_model=128, batch_first=True)

    def forward(self, batch):
        x, y = batch
        output = self.src_embedding(x)
        return output

class LitTranslator(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.model = TranslateModel(src_vocab_size, tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.init_dataloader(Seq2SeqDataset("./output/linq-sample.csv"), 2)

    def _calculate_loss(self, batch, mode="train"):
        x_hat, y, x_lens, y_lens = batch
        loss = self.criterion(x_hat, y)
        info(f" {x_hat.shape=} {y.shape=} {loss.shape=}")
        self.log("%s_loss" % mode, loss)
        return loss

def main():
    start_train(LitTranslator, device='cpu',
                max_epochs=100,
                src_vocab_size=LINQ_VOCAB_SIZE,
                tgt_vocab_size=TSQL_VOCAB_SIZE)

if __name__ == "__main__":
    main()