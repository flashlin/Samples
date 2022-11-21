import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from preprocess_data import pad_sequence, convert_translation_file_to_csv


def convert_seq_to_target(src_values, tgt_values, max_input_seq_length, eos_value: int = 2):
    src_length = len(src_values)
    if src_length < max_input_seq_length:
        src_values = pad_sequence(src_values, eos_value, max_input_seq_length)
        src_length = max_input_seq_length
    tgt_values = pad_sequence(tgt_values, eos_value, max_input_seq_length)
    data_x = []
    data_y = []
    for i in range(0, src_length - max_input_seq_length):
        seq_in = src_values[i:i + max_input_seq_length]
        output_char = tgt_values[i]
        data_x.append(seq_in)
        data_y.append(output_char)
    return data_x, data_y


def validate_size(alist, max_index):
    alist = np.array(alist)
    assert (alist <= max_index).all(), "target: {} invalid ".format(alist)


def comma_str_to_array(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.float16))


class Linq2TSqlDataset(Dataset):
    def __init__(self, csv_file_path):
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        self.df_features = comma_str_to_array(df['features'])
        self.df_labels = comma_str_to_array(df['labels'])
        self.features = torch.tensor(self.df_features, dtype=torch.long)
        self.labels = torch.tensor(self.df_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def create_dataloader(self, batch_size=32):
        train_size = int(0.8 * len(self))
        val_size = len(self) - train_size
        train_data, val_data = random_split(self, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=batch_size)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        return train_loader, val_loader


if __name__ == '__main__':
    convert_translation_file_to_csv()
