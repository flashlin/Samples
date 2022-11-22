import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from common.io import info
from utils.linq_tokenizr import linq_encode
from utils.tokenizr import PAD_TOKEN_VALUE
from utils.tsql_tokenizr import tsql_encode


class TranslationFileTextIterator:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, "r", encoding='UTF-8') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 0:
                    src = line
                else:
                    tgt = line
                    yield src, tgt



class TranslationFileIterator:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, "r", encoding='UTF-8') as f:
            for idx, line in enumerate(f):
                if idx % 2 == 0:
                    linq_values = linq_encode(line)
                else:
                    sql_values = tsql_encode(line)
                    yield linq_values, sql_values


def int_list_to_str(alist):
    return ','.join([str(n) for n in alist])


def convert_translation_file_to_csv(txt_file_path: str = "../data/linq-sample.txt",
                                    output_file_path: str = "./output/linq-sample.csv",
                                    ):
    file_iter = TranslationFileIterator(txt_file_path)
    with open(output_file_path, "w", encoding='utf-8') as csv:
        csv.write('src\ttgt\n')
        for src_values, tgt_values in file_iter:
            csv.write(int_list_to_str(src_values))
            csv.write('\t')
            csv.write(int_list_to_str(tgt_values))
            csv.write('\n')


def pad_seq(seq, fill_value, max_length):
    return seq + [fill_value] * (max_length - len(seq))


def pad_array(arr, fill_value, max_length, d_type=np.long):
    arr_len = len(arr)
    new_arr = np.pad(arr, (0, max_length - arr_len), 'constant', constant_values=fill_value)
    return np.array(new_arr, dtype=d_type)


def comma_str_to_array(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.float16))


def pad_data_loader(dataset, batch_size=32, **kwargs):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=pad_collate, **kwargs)


def df_to_values(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.long))


class Seq2SeqDataset(Dataset):
    def __init__(self, csv_file_path):
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.features = df_to_values(df['src'])
        self.labels = df_to_values(df['tgt'])

    def __len__(self):
        return len(self.features)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        src = torch.tensor(self.features[idx])
        tgt = torch.tensor(self.labels[idx])
        return src, tgt

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size)
        val_loader = pad_data_loader(val_data, batch_size=batch_size)
        return train_loader, val_loader


class TranslationDataset(Dataset):
    def __init__(self, csv_file_path):
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.src = df_to_values(df['src'])
        self.tgt = df_to_values(df['tgt'])

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.src[idx]
        tgt = self.tgt[idx]
        max_len = max(len(src), len(tgt))
        enc_input = torch.tensor(pad_array(src[1:-1], PAD_TOKEN_VALUE, max_len), dtype=torch.long)
        dec_input = torch.tensor(pad_array(tgt[:-1], PAD_TOKEN_VALUE, max_len), dtype=torch.long)
        dec_output = torch.tensor(pad_array(tgt[1:], PAD_TOKEN_VALUE, max_len), dtype=torch.long)
        return enc_input, dec_input, dec_output

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = TranslationDataset.pad_data_loader(train_data, batch_size=batch_size)
        # train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = TranslationDataset.pad_data_loader(val_data, batch_size=batch_size)
        # val_loader = DataLoader(val_data, batch_size=batch_size)
        return train_loader, val_loader

    @staticmethod
    def pad_data_loader(dataset, batch_size=32, **kwargs):
        return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=TranslationDataset.pad_collate, **kwargs)

    @staticmethod
    def pad_collate(batch):
        (enc_input, dec_input, dec_output) = zip(*batch)
        # enc_input_lens = TranslationDataset.get_lens(enc_input)
        # dec_input_lens = TranslationDataset.get_lens(dec_input)
        # dec_input_lens = TranslationDataset.get_lens(dec_output)
        enc_input_pad = torch.nn.utils.rnn.pad_sequence(enc_input, batch_first=True, padding_value=PAD_TOKEN_VALUE)
        dec_input_pad = torch.nn.utils.rnn.pad_sequence(dec_input, batch_first=True, padding_value=PAD_TOKEN_VALUE)
        dec_output_pad = torch.nn.utils.rnn.pad_sequence(dec_output, batch_first=True, padding_value=PAD_TOKEN_VALUE)
        return enc_input_pad, dec_input_pad, dec_output_pad

    @staticmethod
    def get_lens(batch):
        lens = [len(row) for row in batch]
        return lens


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, x_lens, y_lens
