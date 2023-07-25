import numpy as np
import torch
from torch import nn as nn


class CharRNN(nn.Module):
    """
    input_size: ascii 0-127 = 128 特徵大小
    output_size: 分類大小
    """

    def __init__(self, input_size, output_size, hidden_size=8, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        :param x: tensor([
            [[129, 115, 101, 108, 101], [ 99, 116, 130,   0,   0]],
            [[129, 105, 100, 130,   0], [  0,   0,   0,   0,   0]]
        ])
        :return:
            tensor([[0.1454, 0.1765, 0.3148], [0.1416, 0.1781, 0.3188]], grad_fn=<AddmmBackward0>)
        """
        batch_size = x.shape[0]
        x = x.view(-1, batch_size)
        # input shape: [sequence_length, batch_size]
        embedded = self.embedding(x)  # shape: [sequence_length, batch_size, hidden_size]

        # Pass the embedded characters through LSTM
        output, (hidden, _) = self.lstm(embedded)  # output shape: [sequence_length, batch_size, hidden_size]
        # Take the output from the final time step
        output = output[-1, :, :]  # shape: [batch_size, hidden_size]

        # Pass final output through fc layer to get final output
        output = self.fc(output)  # shape: [batch_size, output_size]
        # tensor([[-0.0600, -0.3548,  0.1297],
        #         [ 0.1664, -0.1048,  0.0739],
        #         [ 0.2256, -0.1174,  0.1802],
        #         [ 0.0966, -0.0832,  0.1400],
        #         [ 0.0743, -0.1486,  0.1574]], grad_fn=<AddmmBackward0>)
        # 輸出一個大小為 [5, 3] 的 tensor，
        # 其中5表示你有5個輸入樣本，而3則對應於模型嘗試進行分類的3個類別。
        # 每個張量是該輸入在每個類別上的未經歸一化的預測分數（也被稱為 logits）
        return output


def word_to_chunks(word: str, max_len, start_tag=129, end_tag=130):
    # Convert string to ASCII and add start and end tags
    ascii_vals = [start_tag] + [ord(c) for c in word] + [end_tag]

    # If string length is more than max_len, break it down into chunks of max_len
    if len(ascii_vals) > max_len:
        chunks = [ascii_vals[i:i + max_len] for i in range(0, len(ascii_vals), max_len)]
    else:
        chunks = [ascii_vals]

    # 使用 0 进行填充以确保所有的输入序列都有相同的长度
    padded_chunks = [np.pad(chunk, (0, max_len - len(chunk)), mode='constant') for chunk in chunks]
    return padded_chunks


def pad_array(a_list, max_len, constant_value=0):
    new_list = list(a_list)
    if len(new_list) < max_len:
        padding_length = max_len - len(new_list)
        for _ in range(padding_length):
            new_list.append(constant_value)
    return new_list


def alist_to_chunks(a_list, max_len):
    # If string length is more than max_len, break it down into chunks of max_len
    if len(a_list) > max_len:
        chunks = [a_list[i:i + max_len] for i in range(0, len(a_list), max_len)]
    else:
        chunks = [a_list]

    # 使用 0 进行填充以确保所有的输入序列都有相同的长度
    # padded_chunks = [np.pad(chunk, (0, max_len - len(chunk)), mode='constant') for chunk in chunks]
    padded_chunks = [pad_array(chunk, max_len) for chunk in chunks]
    return padded_chunks


# def padding_alist_chunks_list(alist_chunks_list):
#     data_np = np.array([item for chunks in alist_chunks_list for item in chunks])
#     return data_np


def pad_sequence_list(a_list, max_seq_len):
    result = []
    for sequence in a_list:
        new_sequence = pad_array(sequence, max_seq_len)
        result.append(new_sequence)
    return result


def pad_chunks_list(a_list, max_list_len):
    result = []
    for sequence in a_list:
        sub_list = list(sequence)
        if len(sub_list) < max_list_len:
            sub_list.append(pad_array([], len(sub_list[0])))
        result.append(sub_list)
    return result


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        """
            使用多頭注意力機制, 我們需要決定使用多少頭
            在這種情況下, 我們有 3 個特徵，所以最多可以用 num_heads=3 個頭來處理
            注意hidden_size 隱藏層大小需要能被頭的數量整除
        """
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size // num_heads

        # Define the attention layers for each head
        self.attention_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(num_heads)
        ])

    def forward(self, x):
        """
        :param x: (seq_len, batch_size, hidden_size)
        :return: (seq_len, batch_size, num_heads * 倍數)
        """
        # print(f"mul {x.shape=}")
        # Split the last dimension into (num_heads, hidden_size)
        x = x.view(x.shape[0], x.shape[1], self.num_heads, self.hidden_size)
        # shape: [sequence_length, batch_size, num_heads, hidden_size]

        # Apply attention mechanism for each head
        heads = []
        for i, attention in enumerate(self.attention_heads):
            out = x[:, :, i, :]  # shape: [sequence_length, batch_size, hidden_size]
            weights = attention(out)  # shape: [sequence_length, batch_size, 1]
            weights = torch.nn.functional.softmax(weights, dim=0)  # shape: [sequence_length, batch_size, 1]
            out = out * weights  # shape: [sequence_length, batch_size, hidden_size]
            out = out.sum(dim=0)  # shape: [batch_size, hidden_size]
            heads.append(out)

        # Concatenate all the heads' outputs
        #x = torch.cat(heads, dim=-1)  # shape: [batch_size, hidden_size * num_heads]

        x = torch.stack(heads, dim=0)  # shape: [num_heads, batch_size, sequence_length, hidden_size]
        x = x.permute(2, 1, 0, 3)  # shape: [sequence_length, batch_size, num_heads, hidden_size]
        x = x.contiguous().view(x.shape[0], x.shape[1],
                                -1)  # shape: [sequence_length, batch_size, num_heads * hidden_size]

        return x


def chunks_to_tensor(chunks):
    input_data = torch.LongTensor(chunks)
    return input_data


def get_probability(output):
    """
        output:
            Tensor([ [float,...], [float,...] ]): A tensor of shape (batch_size, num_classes).
            ex tensor([[0.1454, 0.1765, 0.3148], [0.1416, 0.1781, 0.3188]], grad_fn=<AddmmBackward0>)
        Returns:
            list[int]: A list of the class with the highest score for each input in the batch.
            ex [2, 2]
    """
    # 应用softmax函数来计算每个类别的概率
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    # return predicted_class.item()
    return predicted_class.tolist()


class WordRNN(nn.Module):
    """
    output_size: 分類大小
    """

    def __init__(self, output_size, max_pad_word_len=20, hidden_size=9, n_layers=1):
        super(WordRNN, self).__init__()
        self.char_rnn = CharRNN(input_size=256 + 2,
                                output_size=output_size,
                                hidden_size=hidden_size,
                                n_layers=n_layers)
        self.attn = MultiHeadAttention(hidden_size=3,
                                       num_heads=output_size)

    def forward(self, x):
        lstm_outputs = self.char_rnn(x)
        lstm_outputs = lstm_outputs.unsqueeze(0)
        attn_outputs = self.attn(lstm_outputs)
        return get_probability(attn_outputs)


def padding_row_array_list(row_array_list, max_seq_len):
    """
    :param row_array_list:
        [[array([129, 115, 101]), array([ 99, 116, 130)], [array([129, 105, 100)]]
    :param max_seq_len:

    :return:
        [[array([129, 115, 101]), array([ 99, 116, 130)], [array([129, 105, 100), array([0., 0., 0.)]]
    """
    max_length = 0
    for row in row_array_list:
        max_length = max(len(row), max_length)
    new_row_array_list = []
    for row in row_array_list:
        if len(row) < max_length:
            zero_array = [np.zeros(max_seq_len)] * (max_length-len(row))
            new_row_array_list.append(row + zero_array)
        else:
            new_row_array_list.append(row)
    return new_row_array_list
