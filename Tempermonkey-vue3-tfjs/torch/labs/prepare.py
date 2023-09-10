import numpy as np
import torch

from common.io import info
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from preprocess_data import pad_sequence
from utils.linq_translation_data import TranslationFileIterator


def convert_translation_file_to_csv():
    file_iter = TranslationFileIterator("../data/linq-sample.txt")
    src_max_length = 0
    tgt_max_length = 0
    for linq_values, tsql_values in file_iter:
        src_max_length = max(len(linq_values), src_max_length)
        tgt_max_length = max(len(tsql_values), tgt_max_length)
    file_iter = TranslationFileIterator("../data/linq-sample.txt")
    max_length = max(src_max_length, tgt_max_length)
    info(f" translation {src_max_length=} {tgt_max_length=} {max_length=}")
    with open(r"../output/linq-sample.csv", "w", encoding='utf-8') as csv:
        csv.write('linq\ttsql\n')
        for linq_values, tsql_values in file_iter:
            linq_values_padded = pad_sequence(linq_values, 0, src_max_length)
            csv.write(','.join(str(x) for x in linq_values_padded))
            csv.write('\t')
            tsql_values_padded = pad_sequence(tsql_values, 0, tgt_max_length)
            csv.write(','.join(str(x) for x in tsql_values_padded))
            csv.write('\n')

def dataframe_to_array(df):
    return df.map(lambda l: np.array([int(n) for n in l.split(',')], dtype=np.float16))

class Linq2TSqlDataset(Dataset):
    def __init__(self, csv_file_path):
        self.df = df = pd.read_csv(csv_file_path, sep='\t')
        # extract labels
        self.df_linq_values = dataframe_to_array(df['linq'])
        self.df_labels = dataframe_to_array(df['tsql'])
        # drop non numeric columns to make tutorial simpler, in real life do categorical encoding
        # self.df = df.drop(columns=['Type', 'Color', 'Spectral_Class'])
        # convert to torch dtypes
        self.features = torch.tensor(self.df_linq_values).long()
        self.labels = torch.tensor(self.df_labels).long()

    def __len__(self):
        return len(self.features)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_dataloader(csv_file_path):
    ds = Linq2TSqlDataset(csv_file_path)
    dl = DataLoader(ds, batch_size=32, num_workers=2, shuffle=True, drop_last=True)


def create_data_loader(batch_size=32):
    # dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    csv_file_path = "../output/linq-sample.csv"
    dataset = Linq2TSqlDataset(csv_file_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader

if __name__ == '__main__':
    convert_translation_file_to_csv()
