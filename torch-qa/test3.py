import math
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random

from data_utils import pad_list
from sql_network import SqlTrain2Dataset, pad_collate_fn, SqlTrainDataset, sql_to_value, label_value_to_obj, \
    decode_label, label_to_value, key_dict
from tsql_tokenizr import tsql_tokenize


class UniversalTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(UniversalTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)
        self.layers = nn.ModuleList(
            [UniversalTransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoded = self.positional_encoding(embedded)

        for layer in self.layers:
            positional_encoded = layer(positional_encoded)

        output = self.out(positional_encoded)
        return output


class UniversalTransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(UniversalTransformerLayer, self).__init__()
        self.self_attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)

    def forward(self, x):
        self_attention_output = self.self_attention(x, x, x)
        residual_output = x + self_attention_output
        feed_forward_output = self.feed_forward(residual_output)
        output = residual_output + feed_forward_output
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.head_dim = hidden_size // num_heads

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        output = self.out_linear(attention_output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# 創建3筆訓練資料
dataset = SqlTrainDataset("./train_data/sql.txt", max_seq_len=100)
train_loader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate_fn)

# 初始化模型
input_vocab_size = 10000
output_vocab_size = 10000
hidden_size = 32
num_layers = 3
num_heads = 4
model = UniversalTransformer(input_vocab_size, hidden_size, num_layers, num_heads)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

pth_file = './models/test3.pth'
if os.path.exists(pth_file):
    model.load_state_dict(torch.load(pth_file))

def train_model():
    # 訓練模型
    num_epochs = 100
    best_loss = 1000
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            inputs = torch.LongTensor(inputs)
            outputs = model(inputs)
            outputs_flattened = outputs.view(-1, output_vocab_size)
            targets_flattened = targets.view(-1)
            loss = criterion(outputs_flattened, targets_flattened)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), pth_file)
            print(f"save {best_loss}")
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# train_model()

model.eval()  # 進入推斷模式
# 定義推斷函數
def inference(model, input_seq):
    with torch.no_grad():
        input_seq = torch.LongTensor(input_seq).unsqueeze(0)  # 轉換成二維 LongTensor
        outputs = model(input_seq)
        outputs_flattened = outputs.view(-1, output_vocab_size)
        _, predicted_indices = torch.max(outputs_flattened, dim=1)
        predicted_indices = predicted_indices.numpy()
        return predicted_indices

# 測試推斷
sql = "select id, addr, birth from p"
sql_tokens = tsql_tokenize(sql)
s = [token.text for token in sql_tokens]
print(f"{s=}")
input_seq = sql_to_value(sql)
label = {
    'type': 'select',
    'columns': [(1, 2), (1, 4), (1, 6)],
    'froms': [8]
}
label_value = label_to_value(label)
print(f"{input_seq}")
print(f"{label_value}")
label = label_value_to_obj(label_value)
tgt = decode_label(label, sql)
print(f"{tgt}")
print("---------------------------------")

input_seq = pad_list([key_dict['<s>']] + input_seq, 100)
predicted_indices = inference(model, input_seq)
print(predicted_indices)

end_idx = np.where(predicted_indices == 9999)[0][0]
sliced_data = predicted_indices[1:end_idx]
print(f"{sliced_data=}")

label = label_value_to_obj(sliced_data)
tgt = decode_label(label, sql)
print(f"{tgt=}")
