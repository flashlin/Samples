import string

import numpy as np
import torch
import torch.nn as nn

from common.io import info
from ml.data_utils import pad_list
from ml.lit import BaseLightning
from ml.trans_linq2tsql import LinqToSqlVocab
from utils.data_utils import create_char2index_map, create_index2char_map
from utils.stream import StreamTokenIterator, Token, EmptyToken, read_identifier_token, reduce_token_list, \
    read_token_until, read_double_quote_string_token, read_spaces_token, read_symbol_token, read_float_number_token, \
    read_number_token


class ProgramLangVocab:
    def __init__(self):
        printable = ['<pad>', '<bos>', '<eos>'] + [char for char in string.printable]
        spec_symbols = '< > <= >= != <> == && || += -= /= *= %= ' \
                       '( ) { } [ ] . ` @ # $ % ^ & * - + = _ | \\ : " ; \' , . / ?'.split(' ')
        printable += spec_symbols
        self.vocab_size = len(printable)
        self.spec_symbols = spec_symbols
        self.printable = printable
        self.char2index = create_char2index_map(printable)
        self.index2char = create_index2char_map(printable)
        self.padding_idx = self.get_value('<pad>')
        self.bos_idx = self.get_value('<bos>')
        self.eos_idx = self.get_value('<eos>')

    def get_size(self):
        return self.vocab_size

    def decode_values(self, values: [int]) -> [str]:
        return [self.index2char[idx] for idx in values]

    def decode(self, values: [int]) -> str:
        words = self.decode_values(values)
        words = [word for word in words if not word.startswith('<')]
        return ''.join(words)

    @staticmethod
    def read_variable_token(stream_iter: StreamTokenIterator) -> Token:
        if stream_iter.peek_str(1) != '@':
            return EmptyToken
        at_token = stream_iter.next()
        token = read_identifier_token(stream_iter)
        return reduce_token_list('variable', [at_token, token])

    @staticmethod
    def read_spec_identifier_token(stream_iter: StreamTokenIterator) -> Token:
        if stream_iter.peek_str(1) != '[':
            return EmptyToken
        start_token = stream_iter.next()
        ident = read_token_until(stream_iter, ']')
        end_token = stream_iter.next()
        return reduce_token_list(Token.Identifier, [start_token, ident, end_token])

    def parse_to_tokens(self, line) -> [Token]:
        stream_iter = StreamTokenIterator(line)
        buff = []
        while not stream_iter.is_done():
            token = LinqToSqlVocab.read_spec_identifier_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = LinqToSqlVocab.read_variable_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_double_quote_string_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_spaces_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_symbol_token(stream_iter, self.spec_symbols)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_identifier_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_float_number_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            token = read_number_token(stream_iter)
            if token != EmptyToken:
                buff.append(token)
                continue
            raise Exception(f'parse "{stream_iter.peek_str(10)}" fail')
        return buff

    def tokenize_to_words(self, text: str) -> [str]:
        tokens = self.parse_to_tokens(text)
        words = [token.text if token.type != Token.Spaces else ' ' for token in tokens]
        return words

    def get_value(self, char: str) -> int:
        return self.char2index[char]

    def read_text_file_to_words(self, text_file):
        all_tokens = []
        with open(text_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                tokens = self.tokenize_to_words(line)
                all_tokens += tokens
        return all_tokens


class TextClassifier(nn.Module):
    def __init__(self, vocab=ProgramLangVocab(), num_class=3, embedding_dim=1024,
                 hidden_dim=512, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 建立嵌入層
        # self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        self.embedding = TextEmbeddingModel(vocab=vocab, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True,
                            dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(embedding_dim, num_class)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, text):
        # 通過嵌入層，將文字轉換成特徵
        embedded = self.embedding(text)

        output, (hidden, cell) = self.lstm(embedded)
        # hidden = [batch size, n_layers * num directions, hid dim]
        # cell = [batch size, n_layers * num directions, hid dim]
        # 連線最後的正向和反向隱狀態
        hidden = torch.cat((hidden[-2, :], hidden[-1, :]), dim=0)
        # hidden = [batch size, hid dim * num directions]

        # 通過全連接層，將特徵轉換成預測結果
        prediction = self.fc(hidden)
        # info(f" {embedded.shape=} {prediction.shape=}")
        return prediction

    def calculate_loss(self, batch):
        logits, y = batch
        logits = logits.unsqueeze(0)
        y = y.unsqueeze(0)
        return self.loss_fn(logits, y)

    def infer(self, text):
        prediction = self(text)
        return prediction.argmax(0)


class TextEmbeddingModel(nn.Module):
    def __init__(self, vocab, hidden_dim=512, embedding_dim=10):
        super().__init__()
        self.word_embedding = WordEmbeddingModel(vocab=vocab, hidden_dim=hidden_dim, embedding_dim=embedding_dim)

    def forward(self, str_list):
        device = next(self.parameters()).device
        str_tensor = [self.word_embedding(word) for word in str_list]
        str_tensor = torch.stack(str_tensor, dim=0).to(device)
        return str_tensor


class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab, hidden_dim=512, embedding_dim=10):
        super().__init__()
        self.vocab = vocab
        self.merge = MergeModel(1, hidden_dim, embedding_dim)

    def forward(self, input_word):
        input_seq = [self.vocab.char2index[c] for c in input_word]
        input_seq = torch.tensor(input_seq, dtype=torch.float).unsqueeze(dim=1)  # [[1],[2],[3]]
        # 將多個單字向量組合起來
        output = self.merge(input_seq)
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
        last = lstm_out[-1]
        output = self.linear(last)
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


class CBOW(torch.nn.Module):
    def __init__(self, vocab, embedding_dim):
        super().__init__()
        self.vocab = vocab
        # out: 1 x emdedding_dim
        self.embeddings = nn.Embedding(vocab.get_size(), embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.activation_function1 = nn.ReLU()

        # out: 1 x vocab_size
        self.linear2 = nn.Linear(128, vocab.get_size())
        self.activation_function2 = nn.LogSoftmax(dim=-1)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = self.linear1(embeds)
        out = self.activation_function1(out)
        out = self.linear2(out)
        out = self.activation_function2(out)
        return out

    def get_word_emdedding(self, word):
        word = torch.tensor([self.vocab.char2int[word]])
        return self.embeddings(word).view(1, -1)


class LitTransformer2(BaseLightning):
    def __init__(self):
        super().__init__()
        vocab = ProgramLangVocab()
        self.model = TextClassifier(vocab=vocab, num_class=vocab.get_size())
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
        text_to_indices = self.vocab.encode_to_tokens(text)

        src = torch.tensor([text_to_indices]).type(torch.long).to(device)
        src = pad_batch_sequence(src, 200).type(torch.long).to(device)
        self.model.eval()
        logits = self.model(src, src)
        logits = logits.squeeze(0).argmax(1).tolist()
        return self.vocab.decode(logits)
