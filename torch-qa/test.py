import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_running_list


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
                result.append((new_seq, seq[index+1]))
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
#best_model_path = "best_model.pt"
#torch.save(model.state_dict(), best_model_path)

input_sequence = [1, 2, 3]
input_tensor = torch.as_tensor(input_sequence, dtype=torch.float32).unsqueeze(0)
prediction = model(input_tensor)
print(prediction)







###########################
import torch.optim as optim

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        lstm_output, hidden = self.lstm(x, hidden)
        output_x = self.fc(lstm_output)
        return output_x, hidden

# 建立 LSTM 模型
input_size = 1
hidden_size = 64
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)


#
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#
train_data = torch.tensor([[1.0], [2.0], [3.0]])
train_labels = torch.tensor([[5.0], [6.0], [7.0], [8.0], [9.0]])

# 訓練模型
num_epochs = 1000
model.train()
for epoch in range(num_epochs):
    # 初始化隱藏狀態
    hidden = None
    optimizer.zero_grad()
    outputs, hidden = model(train_data.unsqueeze(1), hidden)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    # 每 100 次迭代輸出一次訓練損失
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')



# 加載訓練好的權重（這裡假設你已經訓練好了模型）
# model.load_state_dict(torch.load('best_model.pt'))

# 模型預測
input_sequence = torch.tensor([[1.0], [2.0], [3.0]])  # 輸入序列長度為 3
output_sequence = []
with torch.no_grad():
    hidden = None
    for i in range(10):  # 預測輸出序列長度為 10
        output, hidden = model(input_sequence.unsqueeze(1), hidden)
        output_sequence.append(output[-1, 0].item())
        # 將預測的輸出添加到輸入序列中，用於下一個時間步的預測
        input_sequence = torch.tensor([[output_sequence[-1]]])

print("Predicted Output Sequence:", output_sequence)

