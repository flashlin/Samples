import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_running_list, pad_list, overlap_split_list


###########################
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.query_linear = nn.Linear(input_dim, hidden_dim)
        self.key_linear = nn.Linear(input_dim, hidden_dim)
        self.value_linear = nn.Linear(input_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Linear projections
        query = self.query_linear(query)
        key = self.key_linear(key)
        value = self.value_linear(value)

        # Reshape to (batch_size, num_heads, seq_len, head_dim)
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores and scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, value)

        # Concatenate attention outputs from all heads and linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.output_linear(attention_output)
        return output



class LstmModel(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, output_vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.attention = MultiHeadAttention(hidden_size, hidden_size, num_heads=4)  # 創建多頭注意力機制
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_output, hidden = self.lstm(embedded, hidden)
        attention_output = self.attention(lstm_output, lstm_output, lstm_output)
        # 對注意力機制輸出進行全局平均池化
        attention_output = attention_output.mean(dim=1)
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
    ([2, 3, 4], [4, 5, 6, 7]),
]

max_seq_len = 2

def prepare_train_data(data):
    for input, label in data:
        #input = pad_list(input, max_len=max_seq_len)
        #label = pad_list(label, max_len=max_seq_len)
        inputs = overlap_split_list(input, split_length=max_seq_len, overlap=1)
        labels = overlap_split_list(label, split_length=max_seq_len, overlap=1)
        for new_input, new_label in zip(inputs, labels):
            yield new_input, new_label


train_data = []
for input, label in prepare_train_data(data):
    train_data.append((input, label))


dataset = MyDataset(train_data)
loader = DataLoader(dataset, batch_size=2)


class Seq2SeqModel:
    def __init__(self):
        self.output_vocab_size = 10
        self.model = model = LstmModel(input_vocab_size=1000,
                                       embedding_dim=100,
                                       hidden_size=64,
                                       output_vocab_size=self.output_vocab_size)
        # 定義損失函數和優化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def train(self):
        num_epochs = 1
        model = self.model
        optimizer = self.optimizer
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in loader:
                hidden = None
                optimizer.zero_grad()
                padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
                outputs, hidden = model(padded_inputs, hidden)
                outputs = outputs.view(-1, self.output_vocab_size)
                labels = labels.view(-1)
                # 交叉熵損失函數 nn.CrossEntropyLoss() 需要 labels 的形狀是 [batch_size], 而不是 [batch_size, sequence_length]
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 每 100 次迭代輸出一次訓練損失
                if epoch % 100 == 0:
                    print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}')

    def infer(self, input_seq):
        input_sequence = torch.tensor(input_seq, dtype=torch.long)
        output_sequence = []
        with torch.no_grad():
            hidden = None
            for i in range(10):  # 預測輸出序列長度為 10
                output, hidden = self.model(input_sequence, hidden)
                _, predicted_indices = torch.max(output, dim=-1)
                last_output = predicted_indices[-1].item()
                output_sequence.append(last_output)
                # 將預測的輸出添加到輸入序列中，用於下一個時間步的預測
                new_input = output_sequence[-3:]
                input_sequence = torch.tensor(new_input, dtype=torch.long)
        return output_sequence

m = Seq2SeqModel()
m.train()
output = m.infer([1, 2, 3])
print("Predicted Output Sequence:", output)
