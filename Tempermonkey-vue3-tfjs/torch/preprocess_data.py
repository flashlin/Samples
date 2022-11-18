import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split


def pad_sequence(l, fill_value, max_length):
    return l + [fill_value] * (max_length - len(l))


def comma_str_to_array(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.float16))


def pad_data_loader(dataset, batch_size=32, **kwargs):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=pad_collate, **kwargs)


class Seq2SeqDataset(Dataset):
    def __init__(self, csv_file_path):
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.df_features = comma_str_to_array(df['features'])
        self.df_labels = comma_str_to_array(df['labels'])
        self.features = torch.tensor(self.df_features, dtype=torch.float32)
        self.labels = torch.tensor(self.df_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = pad_data_loader(train_data, batch_size=batch_size)
        val_loader = pad_data_loader(val_data, batch_size=batch_size)
        return train_loader, val_loader


def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    y_lens = [len(y) for y in yy]
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, x_lens, y_lens
