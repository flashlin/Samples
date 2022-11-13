# import spacy
# from torchtext.datasets import Multi30k
import numpy as np
import torch

from common.csv_utils import CsvWriter, CsvReader
from common.io import info
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

from utils.linq_translation_data import Linq2TSqlTranslationFileIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pad_list(l, item, max_length):
    return l + [item] * (max_length - len(l))

def convert_translation_file_to_csv():
    file_iter = Linq2TSqlTranslationFileIterator("../data/linq-sample.txt")
    src_max_length = 0
    tgt_max_length = 0
    for linq_values, tsql_values in file_iter:
        src_max_length = max(len(linq_values), src_max_length)
        tgt_max_length = max(len(tsql_values), tgt_max_length)
    file_iter = Linq2TSqlTranslationFileIterator("../data/linq-sample.txt")
    max_length = max(src_max_length, tgt_max_length)
    info(f"translation {max_length=}")
    with open(r"./output/linq-sample.csv", "w", encoding='utf-8') as csv:
        csv.write('linq\ttsql\n')
        for linq_values, tsql_values in file_iter:
            linq_values_padded = pad_list(linq_values, 0, max_length)
            csv.write(','.join(str(x) for x in linq_values_padded))
            csv.write('\t')
            tsql_values_padded = pad_list(tsql_values, 0, max_length)
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
        self.dataset = torch.tensor(self.df_linq_values).float()
        self.labels = torch.tensor(self.df_labels).long()

    def __len__(self):
        return len(self.dataset)

    # This returns given an index the i-th sample and label
    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


def create_dataloader(csv_file_path):
    ds = Linq2TSqlDataset(csv_file_path)
    dl = DataLoader(ds, batch_size=32, num_workers=2, shuffle=True, drop_last=True)

# class DataModule(LightningDataModule):
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset)
#
#     def val_dataloader(self):
#         return [DataLoader(self.val_dataset_1), DataLoader(self.val_dataset_2)]
#
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset)
#
#     def predict_dataloader(self):
#         return DataLoader(self.predict_dataset)

# def get_datasets(batch_size=128):
#     # Download the language files
#     spacy_de = spacy.load('de')
#     spacy_en = spacy.load('en')
#
#     # define the tokenizer
#     def tokenize_de(text):
#         return [token.text for token in spacy_de.tokenizer(text)]
#
#     def tokenize_en(text):
#         return [token.text for token in spacy_en.tokenizer(text)]
#
#     # Create the pytext's Field
#     source = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
#     target = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
#
#     # Splits the data in Train, Test and Validation data
#     train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(source, target))
#
#     info(f"{type(train_data)=}")
#     info(f"{train_data=}")
#
#     # Build the vocabulary for both the language
#     source.build_vocab(train_data, min_freq=3)
#     target.build_vocab(train_data, min_freq=3)
#
#     # Create the Iterator using builtin Bucketing
#     train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data),
#                                                                           batch_size=batch_size,
#                                                                           sort_within_batch=True,
#                                                                           sort_key=lambda x: len(x.src),
#                                                                           device=device)
#     return train_iterator, valid_iterator, test_iterator, source, target

if __name__ == '__main__':
    #train_iterator, valid_iterator, test_iterator, source, target = get_datasets(batch_size=256)
    convert_translation_file_to_csv()


def create_data_loader(batch_size=32):
    # dataset = MNIST("", train=True, download=True, transform=transforms.ToTensor())
    csv_file_path = "./output/linq-sample.csv"
    dataset = Linq2TSqlDataset(csv_file_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader
