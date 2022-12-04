import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F
from ml.bpe_tokenizer import SimpleTokenizer
from ml.lit import BaseLightning
from ml.mnt_model import NMTModel
from dataclasses import dataclass
from typing import Callable

from labs.preprocess_data import TranslationFileTextIterator, TranslationDataset, int_list_to_str, \
    df_intstr_to_values, pad_data_loader
from utils.data_utils import pad_array
from utils.linq_tokenizr import linq_tokenize
from utils.stream import Token
from utils.tsql_tokenizr import tsql_tokenize


class MntTokenizer(SimpleTokenizer):
    def __init__(self, tokenize_fn):
        super().__init__(tokenize_fn)
        token_types = [Token.Identifier, Token.String,
                       Token.Number, Token.Spaces, Token.Symbol, Token.Keyword, Token.Operator]
        self.vocab.extend(token_types)
        self.vocab_size, self.encoder, self.decoder = self.calculate_encoder_decoder(self.vocab)
        for token_type in token_types:
            self.cache[token_type] = token_type

    def encode(self, text, add_start_end=False):
        def add_to_bpe_tokens(a_token):
            bpe_tokens.extend([self.encoder[a_token.type]])
            token_text = ''.join(self.byte_encoder[b] for b in a_token.text.encode_tokens('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token_text).split(' '))

        bpe_tokens = []
        tokens = self.tokenize_fn(text)
        for token in tokens:
            add_to_bpe_tokens(token)
        if add_start_end:
            bpe_tokens = [self.bos_idx] + bpe_tokens + [self.eos_idx]
        return bpe_tokens


class TranslationFileEncodeIterator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.linq_tk = MntTokenizer(linq_tokenize)
        self.tsql_tk = MntTokenizer(tsql_tokenize)

    def __iter__(self):
        for src, tgt in TranslationFileTextIterator(self.file_path):
            src_tokens = self.linq_tk.encode(src, add_start_end=True)
            tgt_tokens = self.tsql_tk.encode(tgt, add_start_end=True)
            yield src_tokens, len(src_tokens), tgt_tokens, len(tgt_tokens)


def write_train_data():
    with open('./output/linq-sample.csv', "w", encoding='UTF-8') as csv:
        csv.write('src\tsrc_len\ttgt\ttgt_len\n')
        for src, src_len, tgt, tgt_len in TranslationFileEncodeIterator('../data/linq-sample.txt'):
            csv.write(int_list_to_str(src))
            csv.write('\t')
            csv.write(f'{src_len}')
            csv.write('\t')
            csv.write(int_list_to_str(tgt))
            csv.write('\t')
            csv.write(f'{tgt_len}')
            csv.write('\n')


def pad_data_loader(dataset, batch_size, padding_idx):
    def pad_collate(batch):
        (src, src_len, tgt, tgt_len) = zip(*batch)
        src_pad = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=padding_idx)
        tgt_pad = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=padding_idx)
        return src_pad, src_len, tgt_pad, tgt_len

    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=pad_collate, pin_memory=True)


class TranslationDataset(Dataset):
    def __init__(self, csv_file_path, padding_idx):
        self.padding_idx = padding_idx
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.src = df_intstr_to_values(df['src'])
        self.src_len = df['src_len']
        self.tgt = df_intstr_to_values(df['tgt'])
        self.tgt_len = df['tgt_len']

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        src_len = self.src_len[idx]
        tgt = self.tgt[idx]
        tgt_len = self.tgt_len[idx]
        max_len = max(src_len, tgt_len)
        src = torch.tensor(pad_array(src, self.padding_idx, max_len), dtype=torch.long)
        tgt = torch.tensor(pad_array(tgt, self.padding_idx, max_len), dtype=torch.long)
        # dec_output = torch.tensor(pad_array(tgt[1:], self.padding_idx, max_len), dtype=torch.long)
        return src, src_len, tgt, tgt_len

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size, padding_idx=self.padding_idx)
        val_loader = pad_data_loader(val_data, batch_size=batch_size, padding_idx=self.padding_idx)
        return train_loader, val_loader


@dataclass(frozen=True)
class TranslateOptions:
    src_vocab_size: int
    src_embedding_size: int
    tgt_vocab_size: int
    tgt_embedding_size: int
    encoding_size: int
    bos_idx: int
    eos_idx: int
    padding_idx: int
    mask_idx: int
    unk_idx: int
    num_epochs: int
    decode_fn: Callable[[list[int]], str]
    train_dataset: Dataset


tk = SimpleTokenizer(None)

translate_options = TranslateOptions(
    src_vocab_size=tk.vocab_size,
    src_embedding_size=128,
    tgt_vocab_size=tk.vocab_size,
    tgt_embedding_size=128,
    encoding_size=400,
    bos_idx=tk.bos_idx,
    eos_idx=tk.eos_idx,
    padding_idx=tk.padding_idx,
    mask_idx=tk.mask_idx,
    unk_idx=tk.unk_idx,
    num_epochs=100,
    decode_fn=tk.decode,
    train_dataset=lambda: TranslationDataset("./output/linq-sample.csv", tk.padding_idx)
)


def normalize_sizes(y_pred, y_true):
    """Normalize tensor sizes
    Args:
        y_pred (torch.Tensor): the output of the model
            If a 3-dimensional tensor, reshapes to a matrix
        y_true (torch.Tensor): the target predictions
            If a matrix, reshapes to be a vector
    """
    if len(y_pred.size()) == 3:
        y_pred = y_pred.contiguous().view(-1, y_pred.size(2))
    if len(y_true.size()) == 2:
        y_true = y_true.contiguous().view(-1)
    return y_pred, y_true


def sequence_loss(y_pred, y_true, mask_index):
    y_pred, y_true = normalize_sizes(y_pred, y_true)
    return F.cross_entropy(y_pred, y_true, ignore_index=mask_index)


class MntTranslator(BaseLightning):
    def __init__(self, options=translate_options):
        super().__init__()
        self.options = options
        self.model = NMTModel(source_vocab_size=options.src_vocab_size,
                              source_embedding_size=options.src_embedding_size,
                              target_vocab_size=options.tgt_vocab_size,
                              target_embedding_size=options.tgt_embedding_size,
                              encoding_size=options.encoding_size,
                              target_bos_index=options.bos_idx)
        self.criterion = sequence_loss
        self.init_dataloader(options.train_dataset(), 1)
        self.sample_probability = None

    @staticmethod
    def prepare_train_data():
        write_train_data()

    def training_step(self, batch, batch_idx):
        self.sample_probability = (20 + batch_idx) / self.options.num_epochs
        outputs = self(batch)
        loss = self._calculate_loss((outputs, batch), mode="train")
        return loss

    def forward(self, batch):
        src, src_lens, tgt, tgt_lens = batch
        y_pred = self.model(src,
                            src_lens,
                            tgt,
                            sample_probability=self.sample_probability)
        return y_pred, tgt

    def _calculate_loss(self, data, mode="train"):
        (logits, tgt), batch_idx = data
        loss = self.criterion(logits, tgt, self.options.mask_index)
        self.log("%s_loss" % mode, loss)
        return loss

    def infer(self, text):
        tgt_values = self.model.inference(text)
        tgt_text = self.options.decode_fn(tgt_values)
        return tgt_text
