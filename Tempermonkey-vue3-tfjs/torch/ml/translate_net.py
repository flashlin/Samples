import torch
from torch import nn
from torch.utils.data import Dataset, random_split
import pandas as pd

from common.io import info
from ml.data_utils import pad_list
from ml.lit import BaseLightning, PositionalEncoding
from ml.mnt_net import pad_data_loader
from ml.model_utils import reduce_dim
from utils.data_utils import df_intstr_to_values


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, padding_idx, word_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, word_dim)
        self.pos_emb = PositionalEncoding(word_dim, dropout=0.1)
        self.transformer = nn.Transformer(d_model=word_dim,
                                          nhead=8,  # default:8
                                          num_encoder_layers=6,  # default:6
                                          num_decoder_layers=6,
                                          dropout=0.1,
                                          batch_first=True)
        self.predictor = nn.Linear(word_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, x, y):
        outputs = self.transform(x, y)
        outputs = self.predictor(outputs)
        return outputs

    def transform(self, x, y):
        src_key_padding_mask = self.get_key_padding_mask(x)
        tgt_key_padding_mask = self.get_key_padding_mask(y)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(y.size(-1)).to(x.device)
        x = self.embedding(x)
        x = self.pos_emb(x)
        y = self.embedding(y)
        y = self.pos_emb(y)
        outputs = self.transformer(x, y,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=src_key_padding_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask
                                   )
        return outputs

    def calculate_loss(self, x_hat, y):
        n_tokens = (y != self.padding_idx).sum()

        x_hat = x_hat.contiguous().view(-1, x_hat.size(-1))
        y = y.contiguous().view(-1)

        loss = self.loss_fn(x_hat, y) / n_tokens
        return loss

    def get_key_padding_mask(self, tokens):
        key_padding_mask = tokens == self.padding_idx
        # # key_padding_mask = self.transformer.generate_square_subsequent_mask(tokens.size())
        # key_padding_mask = torch.zeros(tokens.size()).type(torch.bool)
        # key_padding_mask[tokens == self.padding_idx] = True
        return key_padding_mask


class TranslateListDataset(Dataset):
    def __init__(self, translation_list, vocab):
        self.vocab = vocab
        self.data = translation_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src = self.vocab.encode_to_tokens(src)
        tgt = self.vocab.encode_to_tokens(tgt)
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return src, len(src), tgt, len(tgt)

    def create_dataloader(self, batch_size=32):
        vocab = self.vocab
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=vocab.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=vocab.padding_idx)
        return train_loader, val_loader


class TranslateCsvDataset(Dataset):
    def __init__(self, translation_csv_file_path, vocab):
        self.vocab = vocab
        self.df = df = pd.read_csv(translation_csv_file_path, sep='\t')
        self.src = df['src']
        self.tgt = df['tgt']

    def __len__(self):
        return len(self.src)

    def encode_text(self, text):
        buf = self.vocab.encode_to_tokens(text)
        return [self.vocab.bos_idx] + buf + [self.vocab.eos_idx]

    def __getitem__(self, idx):
        max_length = 300
        src = self.encode_text(self.src[idx])
        tgt = self.encode_text(self.tgt[idx])
        src_len = len(src)
        tgt_len = len(tgt)
        src_len = torch.tensor(src_len, dtype=torch.long)
        tgt_len = torch.tensor(tgt_len, dtype=torch.long)
        # src = pad_list(src, max_length)
        # tgt = pad_list(tgt, max_length)
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        return src, src_len, tgt, tgt_len

    def create_dataloader(self, batch_size=32):
        vocab = self.vocab
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=vocab.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=vocab.padding_idx)
        return train_loader, val_loader


def get_translate_file_iter(translate_text_file_path):
    with open(translate_text_file_path, "r", encoding='UTF-8') as f:
        for idx, line in enumerate(f):
            if idx % 2 == 0:
                src = line.rstrip()
            else:
                tgt = line.rstrip()
                yield src, tgt


def convert_translate_file_to_csv(translate_text_file_path, csv_file_path):
    with open(csv_file_path, "w", encoding='UTF-8') as csv:
        csv.write(f"src\ttgt\n")
        for src, tgt in get_translate_file_iter(translate_text_file_path):
            csv.write(f"{src}\t{tgt}\n")


class LiTranslator(BaseLightning):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.model = Seq2SeqTransformer(vocab.get_size(), vocab.get_value('<pad>'), word_dim=128)

    def forward(self, batch):
        src, src_len, tgt, tgt_len = batch

        tgt_y = tgt[:, 1:]
        tgt = tgt[:, :-1]

        x_hat = self.model(src, tgt)
        return x_hat, tgt_y

    def _calculate_loss(self, data, mode="train"):
        (x_hat, y_hat), batch = data
        loss = self.model.calculate_loss(x_hat, y_hat)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        vocab = self.vocab
        src_values = vocab.encode_to_tokens(text)
        self.model.eval()
        device = next(self.parameters()).device
        src = torch.tensor([src_values], dtype=torch.long).to(device)
        bos = vocab.get_value('<bos>')
        tgt = torch.tensor([[bos]], dtype=torch.long).to(device)
        for i in range(len(src_values)):
            outputs = self.model.transform(src, tgt)
            # 預測結果，因為只需要看最後一個詞，所以取`out[:, -1]`
            last_word = outputs[:, -1]
            predict = self.model.predictor(last_word)
            # 找出最大值的 index
            y = torch.argmax(predict, dim=1)
            # 和之前的預測結果拼接到一起
            tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
            if y == vocab.get_value('<eos>'):
                break

        result = vocab.decode(reduce_dim(tgt).tolist())
        return result
