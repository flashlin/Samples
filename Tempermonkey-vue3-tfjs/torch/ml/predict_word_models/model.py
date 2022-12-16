import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np

from ml.data_utils import create_dataloader
from ml.lit import BaseLightning


def one_hot_encode(arr, n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


class Vocab:
    def __init__(self):
        symbols = '. ( ) [ ] { } = _ + - * / , ? ~ > < ! @ \' " & % `'.split(' ')
        symbols += '1 2 3 4 5 6 7 8 9 0 <unk> <bos> <eos> <pad>'.split(' ') + [' ']
        letters = symbols + list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        chars = tuple(set(letters))
        self.int2char = dict(enumerate(chars))
        self.char2int = {char: index for index, char in self.int2char.items()}

    def encode(self, word):
        return np.array([self.char2int[ch] for ch in word])

    def decode(self, values):
        return ''.join([self.int2char[index] for index in values])


def get_batches(arr, n_seqs, n_steps):
    batch_size = n_seqs * n_steps
    n_batches = len(arr)

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]

    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):

        # The features
        x = arr[:, n:n + n_steps]

        # The targets, shifted by one
        y = np.zeros_like(x)

        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + n_steps]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


class CharRNN(nn.Module):
    def __init__(self, vocab,
                 n_hidden=256,
                 n_layers=2,
                 drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.vocab = vocab
        self.n_chars = vocab.get_size()
        self.lstm = nn.LSTM(vocab.get_size(),
                            n_hidden,
                            n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, len(self.chars))
        self.hidden = None

    def forward(self, x):
        input = F.one_hot(x, num_classes=self.n_chars).type(torch.FloatTensor).unsqueeze(0)
        if self.hidden is None:
            l_output, self.hidden = self.lstm(input)
        else:
            l_output, self.hidden = self.lstm(input, self.hidden)

        l_output = self.dropout(l_output)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        return self.fc(l_output)

    def calculate_loss(self, x, y):
        loss = F.cross_entropy.loss_fn(x.squeeze()[-1:], y[-1:])
        return loss

    def train_step(self, word):
        i = 0
        count = 0
        total_loss = 0
        while i < len(word) - 40:
            seq_len = random.randint(10, 40)
            input, target = word[i:i + seq_len], word[i + 1:i + 1 + seq_len]
            i += seq_len
            output = self(input)
            count += 1
            loss = self.calculate_loss(output, target)
            total_loss += loss
        return total_loss

    def predict(self, char, h=None, top_k=None):
        vocab = self.vocab
        x = np.array([[vocab.char2int[char]]])
        inputs = F.one_hot(x, num_classes=self.n_chars).type(torch.FloatTensor).unsqueeze(0)

        if h is None:
            out, h = self.forward(inputs)
        else:
            h = tuple([each.words for each in h])
            out, h = self.forward(inputs, h)

        p = F.softmax(out, dim=1).data
        if top_k is None:
            top_ch = np.arange(len(self.n_chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())
        return vocab.int2char[char], h


class WordDataset(Dataset):
    def __init__(self, file_path, vocab):
        words = {}
        with open(file_path, mode='r', encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            line_words = vocab.encode_to_texts(line)
            for word in line_words:
                words[word] = 1
        self.words = words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        return word

    def create_dataloader(self, batch_size=32):
        return create_dataloader(self, batch_size)


class WordPredictor(BaseLightning):
    def __init__(self, vocab):
        self.model = CharRNN(vocab)

    def forward(self, batch):
        loss = self.model.train_step(batch)
        return loss

    def _calculate_loss(self, batch, mode="train"):
        loss = batch
        self.log("%s_loss" % mode, loss)
        return loss

