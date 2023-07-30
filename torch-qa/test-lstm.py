import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
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



###########################
class LstmModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        output_x = self.fc(lstm_output)
        return output_x, hidden


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

# 建立模型
vocab_size = 10000
output_vocab_size = 10
model = LstmModel(vocab_size, embedding_dim=100, hidden_size=64, output_vocab_size=output_vocab_size)

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
        outputs, hidden = model(padded_inputs, hidden)
        outputs = outputs.view(-1, output_vocab_size)
        labels = labels.view(-1)
        # 交叉熵損失函數 nn.CrossEntropyLoss() 需要 labels 的形狀是 [batch_size], 而不是 [batch_size, sequence_length]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # 每 100 次迭代輸出一次訓練損失
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}')

# 使用訓練好的模型進行預測
input_sequence = torch.tensor([1, 2, 3], dtype=torch.long)  # 輸入序列長度為 3
output_sequence = []
with torch.no_grad():
    hidden = None
    for i in range(10):  # 預測輸出序列長度為 10
        output, hidden = model(input_sequence, hidden)
        _, predicted_indices = torch.max(output, dim=-1)
        last_output = predicted_indices[-1].item()
        output_sequence.append(last_output)
        # 將預測的輸出添加到輸入序列中，用於下一個時間步的預測
        new_input = output_sequence[-3:]
        # new_input = pad_list(new_input, 3)
        input_sequence = torch.tensor(new_input, dtype=torch.long)

print("Predicted Output Sequence:", output_sequence)
