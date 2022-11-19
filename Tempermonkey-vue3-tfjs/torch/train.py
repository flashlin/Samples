import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from common.io import info
from preprocess_data import Seq2SeqDataset, convert_translation_file_to_csv
from lit import BaseLightning, start_train, PositionalEncoding, CosineWarmupScheduler, MultiHeadAttention
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tokenizr import PAD_TOKEN_VALUE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE

class SeqEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional = PositionalEncoding(d_model)

    def forward(self, x):
        """
        :param x: [long]
        :return: [batch, seq_len, dim]
        """
        output = self.embedding(x)
        output = self.positional(output)
        return output

class TranslateModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_dim=3, num_heads=8):
        super().__init__()
        d_model = d_dim * num_heads
        self.src_embedding = SeqEmbedding(src_vocab_size, d_model)
        self.tgt_embedding = SeqEmbedding(tgt_vocab_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=num_heads, batch_first=True)

    def forward(self, batch):
        src, tgt = batch
        src = src.unsqueeze(0)
        tgt = tgt.unsqueeze(0)
        x_embedding = self.src_embedding(src)
        y_embedding = self.tgt_embedding(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1))
        src_key_padding_mask = TranslateModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TranslateModel.get_key_padding_mask(tgt)
        output = self.transformer(x_embedding, y_embedding,
                              tgt_mask=tgt_mask,
                              src_key_padding_mask=src_key_padding_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        # output = [batch, seq_len, word_dim]
        return output, y_embedding

    @staticmethod
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == PAD_TOKEN_VALUE] = -torch.inf
        return key_padding_mask

class LitTranslator(BaseLightning):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.model = TranslateModel(src_vocab_size, tgt_vocab_size)
        self.criterion = nn.CrossEntropyLoss() #reduction="none")
        self.init_dataloader(Seq2SeqDataset("./output/linq-sample.csv"), 2)

    def _fetch_xy_batch(self, batch):
        x, y, x_lens, y_lens = batch
        return x, y

    def _calculate_loss(self, batch, mode="train"):
        x, y = batch
        x_hat, y_true = x
        loss = self.criterion(x_hat, y_true)
        self.log("%s_loss" % mode, loss)
        return loss

def prepare_train_data():
    print("convert translation file to csv...")
    convert_translation_file_to_csv()
    print("done.")

def main():
    start_train(LitTranslator, device='cpu',
                max_epochs=100,
                src_vocab_size=LINQ_VOCAB_SIZE,
                tgt_vocab_size=TSQL_VOCAB_SIZE)

if __name__ == "__main__":
    prepare_train_data()
    main()