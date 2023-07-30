import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_running_list, pad_list


def prepare_data(data, seq_length):
    def get_next(index, seq):
        if index < len(seq) - 1:
            next = seq[index + 1]
        return next

    result = []
    for seq in data:
        new_seqs = create_running_list(seq, seq_length)
        for index, new_seq in enumerate(new_seqs):
            if index < len(new_seqs) - 1:
                result.append((new_seq, seq[index + 1]))
    return result


data = [[1, 2, 3], [4, 5, 6, 7]]
train_data = prepare_data(data, 3)
print(train_data)


class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        embed_dim = 63
        self.lstm = nn.LSTM(input_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=3)
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        lstm_output, hidden_state = self.lstm(x)
        query = lstm_output
        key = hidden_state[0]
        value = hidden_state[0]
        attn_output, _ = self.attention(query, key, value)
        output_x = self.fc(attn_output)
        return output_x


# 建立訓練資料集
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input, label = self.data[index]
        input_tensor = torch.as_tensor(input, dtype=torch.float32)
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        return input_tensor, label_tensor


dataset = MyDataset(train_data)
# 建立 DataLoader
loader = DataLoader(dataset, batch_size=2)

# 建立模型
model = LSTMAttentionModel(3, 200)

# 定義損失函數和優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 訓練模型
for epoch in range(10):
    for batch in loader:
        inputs, labels = batch
        print(f"{inputs=}")
        predictions = model(inputs)
        loss = loss_fn(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 保存最佳權重
# best_model_path = "best_model.pt"
# torch.save(model.state_dict(), best_model_path)

input_sequence = [1, 2, 3]
input_tensor = torch.as_tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
prediction = model(input_tensor)
print(prediction)

###########################
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


# 建立 LSTM + 詞嵌入模型
class LSTMEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        output_x = self.fc(lstm_output)
        return output_x, hidden


# 建立詞彙表
vocab_size = 10000
embedding_dim = 100

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        input, label = self.data[index]
        input_tensor = torch.as_tensor(input, dtype=torch.long)
        label_tensor = torch.as_tensor(label, dtype=torch.long)
        return input_tensor, label_tensor


data = [
    ([1, 2, 3], [4, 5, 6]),
    ([1, 2, 3], [4, 5, 6]),
    ([1, 2, 3], [4, 5, 6]),
    ([1, 2, 3], [4, 5, 6]),
]

dataset = MyDataset(data)
loader = DataLoader(dataset, batch_size=2)


# # 輸入資料，這裡使用整數編碼表示單詞
# train_data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
# # 輸出標籤，這裡使用整數編碼表示單詞
# train_labels = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)

# 建立模型
hidden_size = 64
output_size = 10
model = LSTMEmbeddingModel(vocab_size, embedding_dim, hidden_size, output_size)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for inputs, labels in loader:
        hidden = None
        optimizer.zero_grad()
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        # padded_inputs = padded_inputs.unsqueeze(0)
        outputs, hidden = model(padded_inputs, hidden)
        print(f"{outputs.shape=}")
        print(f"{labels.shape=}")
        outputs = outputs.view(-1, output_size)
        labels = labels.view(-1)
        # 交叉熵損失函數 nn.CrossEntropyLoss() 需要 labels 的形狀是 [batch_size]，而不是 [batch_size, sequence_length]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 每 100 次迭代輸出一次訓練損失
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用訓練好的模型進行預測
input_sequence = torch.tensor([1, 2, 3], dtype=torch.long)  # 輸入序列長度為 3
output_sequence = []
with torch.no_grad():
    hidden = None
    for i in range(10):  # 預測輸出序列長度為 10
        output, hidden = model(input_sequence, hidden)
        _, predicted_indices = torch.max(output, dim=-1)
        # print(f"{predicted_indices=}")
        # last_output = predicted_indices[-1, 0].item()
        last_output = predicted_indices[-1].item()
        # print(f"{last_output=}")
        output_sequence.append(last_output)
        # 將預測的輸出添加到輸入序列中，用於下一個時間步的預測
        new_input = output_sequence[-3:]
        # print(f"1{new_input=}")
        new_input = pad_list(new_input, 3)
        # print(f"2{new_input=}")
        input_sequence = torch.tensor(new_input, dtype=torch.long)

print("Predicted Output Sequence:", output_sequence)
