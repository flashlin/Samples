import math
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.io import info
from ml.data_utils import pad_list
from ml.lit import BaseLightning
from utils.data_utils import create_char2index_map

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_class):
        super(TextClassifier, self).__init__()
        # 建立嵌入層
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        # 建立全連接層
        self.fc = nn.Linear(embedding_dim, num_class)

    def forward(self, text, offsets):
        # 通過嵌入層，將文字轉換成特徵
        embedded = self.embedding(text, offsets)
        # 通過全連接層，將特徵轉換成預測結果
        return self.fc(embedded)


class TextEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim=512, embedding_dim=10000):
        super().__init__()
        self.word_embedding = WordEmbeddingModel(hidden_dim=hidden_dim, embedding_dim=embedding_dim)

    def forward(self, str_list):
        str_tensor = [self.word_embedding(word) for word in str_list]
        return str_tensor


class WordEmbeddingModel(nn.Module):
    def __init__(self, hidden_dim=512, embedding_dim=10000):
        super().__init__()
        printable = ['<pad>', '<bos>', '<eos>'] + [char for char in string.printable]
        vocab_size = len(printable)
        self.char2index = create_char2index_map(printable)
        self.embedding = MergeModel(1, hidden_dim, 1)

    def forward(self, input_word):
        input_seq = [self.char2index[c] for c in input_word]
        input_seq = torch.tensor(input_seq, dtype=torch.float).unsqueeze(dim=1)  # [[1],[2],[3]]
        # 將多個單字向量組合起來
        output = self.embedding(input_seq)
        # output_vector = np.mean(output.detach().numpy(), axis=0)
        # return torch.mean(output, dim=0)
        return output


class MergeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MergeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the layers of the model
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        :param x: (batch, seq_len, dim)
        :return:
        """
        lstm_out, _ = self.lstm(x)
        output = self.linear(lstm_out[-1])
        return output


def create_tensor(shape):
    return torch.tensor(np.zeros(shape=shape))


def pad_batch_sequence(batch_tensor, max_length):
    new_batch = create_tensor((batch_tensor.size(0), max_length))
    for n in range(batch_tensor.size(0)):
        alist = batch_tensor[n, :].tolist()
        pad_sequence = pad_list(alist, max_length=max_length)
        new_batch[n, :] = torch.tensor(pad_sequence)
    return new_batch


class LitTransformer2(BaseLightning):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        # CrossEntropyLoss((src_len, n_classes), (tgt_len))
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx)

    def forward(self, batch):
        device = next(self.parameters()).device
        src, src_lens, tgt, tgt_lens = batch
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        tgt = pad_batch_sequence(tgt, 200).type(torch.long).to(device)
        return self.model(src, tgt), tgt

    def _calculate_loss(self, batch, batch_idx):
        x_hat, y = batch
        x_hat = x_hat.squeeze(0)
        y_true = y.squeeze(0)
        return self.loss_fn(x_hat, y_true)

    def infer(self, text):
        device = self.get_device()
        text_to_indices = self.vocab.encode(text)

        src = torch.tensor([text_to_indices]).type(torch.long).to(device)
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        self.model.eval()
        logits = self.model(src, src)
        logits = logits.squeeze(0).argmax(1).tolist()
        return self.vocab.decode(logits)
