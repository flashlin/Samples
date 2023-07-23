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

    """
    """
    def forward(self, x):
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

        print(f"model {output=}")

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


def word_chunks_list_to_chunks(word_chunks_list):
    data_np = np.array([item for chunks in word_chunks_list for item in chunks])
    return data_np


class MultiHeadAttention(nn.Module):
    """
        使用多頭注意力機制，我們需要決定使用多少頭。讓我們假設我們使用 3 個頭。
        這意味著我們的隱藏層大小需要能被頭的數量整除。在這種情況下，我們有 3 個特徵，所以可以用 3 個頭來處理。
    """
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size // num_heads

        # Define the attention layers for each head
        self.attention_heads = nn.ModuleList([
            nn.Linear(self.hidden_size, 1) for _ in range(num_heads)
        ])

    def forward(self, x):
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
        x = torch.cat(heads, dim=-1)  # shape: [batch_size, hidden_size * num_heads]

        return x


def chunks_to_tensor(chunks):
    input_data = torch.LongTensor(chunks)
    return input_data


def get_probability(output):
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
        self.char_rnn = CharRNN(input_size=256+2,
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


