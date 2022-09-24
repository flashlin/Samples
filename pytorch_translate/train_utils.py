from collections import Counter
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import math

from torchtext.vocab import vocab
from data_utils import textdata_to_tensor_iter, load_csv_to_dataframe


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


# def train_epoch(model, train_iter, optimizer, device):
#     model.train_df()
#     losses = 0
#     for idx, (src, tgt) in enumerate(train_iter):
#         src = src.to(device)
#         tgt = tgt.to(device)
#         tgt_input = tgt[:-1, :]
#
#         src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
#
#         logits = model(src, tgt_input, src_mask, tgt_mask,
#                        src_padding_mask, tgt_padding_mask, src_padding_mask)
#
#         optimizer.zero_grad()
#
#         tgt_out = tgt[1:, :]
#         loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
#         loss.backward()
#
#         optimizer.step()
#         losses += loss.item()
#     torch.save(model, PATH)
#     return losses / len(train_iter)


def textdata_to_train_data(source_filepath: str, source_vocab, source_tokenizer,
                           target_filepath: str, target_vocab, target_tokenizer):
    source_tensor_iter = textdata_to_tensor_iter(source_filepath, source_vocab, source_tokenizer)
    target_tensor_iter = textdata_to_tensor_iter(target_filepath, target_vocab, target_tokenizer)
    data = []
    for (source_tensor, target_tensor) in zip(source_tensor_iter, target_tensor_iter):
        data.append((source_tensor, target_tensor))
    return data


def create_mask(src, tgt, pad_idx: int, device="gpu"):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def evaluate(model, loss_fn, val_iter, pad_idx, device="gpu"):
    model.eval()
    losses = 0
    for idx, (src, tgt) in (enumerate(val_iter)):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx, device)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)


def build_vocab_from_dataframe(df: pd.DataFrame, column_name: str, tokenizer):
    counter = Counter()
    for text in df[column_name]:
        counter.update(tokenizer(text))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def translate_csv_to_tensor_iter(csv_filepath: str, source_column_name: str, source_tokenizer,
                                 target_column_name: str, target_tokenizer):
    dataframe = load_csv_to_dataframe(csv_filepath)
    source_vocab = build_vocab_from_dataframe(dataframe, source_column_name, source_tokenizer)
    target_vocab = build_vocab_from_dataframe(dataframe, target_column_name, target_tokenizer)
    return source_vocab, target_vocab


def generate_batch(data_batch, bos_idx, eos_idx, pad_idx):
    source_batch, target_batch = [], []
    for (source_item, target_item) in data_batch:
        source_batch.append(torch.cat([torch.tensor([bos_idx]), source_item, torch.tensor([eos_idx])], dim=0))
        target_batch.append(torch.cat([torch.tensor([bos_idx]), target_item, torch.tensor([eos_idx])], dim=0))
    source_batch = pad_sequence(source_batch, padding_value=pad_idx)
    target_batch = pad_sequence(target_batch, padding_value=pad_idx)
    return source_batch, target_batch


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train_epoch(model, loss_fn, train_iter, optimizer, device):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)
