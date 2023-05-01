from torch.optim import SGD
import torch
import torch.nn as nn
import os
import string


class CharDict:
    char_to_index = {}
    index_to_char = {}

    def __init__(self):
        letters = string.ascii_letters + " " + string.digits + string.punctuation + "\0"
        self.scan_sentence(letters)

    def scan_sentence(self, sentence):
        for char in sentence:
            if char not in self.char_to_index:
                index = len(self.char_to_index)
                self.char_to_index[char] = index
                self.index_to_char[index] = char

    def create_train_data(self, data):
        # 計算最大句子長度
        max_length = max(len(sentence) for sentence in data)
        pad_char = '\0'
        # 將數據轉換為張量
        input_data = []
        target_data = []
        for sentence in data:
            input_sentence = [self.char_to_index[char] for char in sentence[:-1]]
            target_sentence = [self.char_to_index[char] for char in sentence[1:]]
            input_data.append(input_sentence)
            target_data.append(target_sentence)
            # 在較短的句子後面填充字符
            input_sentence += [self.char_to_index[pad_char]] * (max_length - len(input_sentence))
            target_sentence += [self.char_to_index[pad_char]] * (max_length - len(target_sentence))
        input_tensor = torch.tensor(input_data)
        target_tensor = torch.tensor(target_data)
        return input_tensor, target_tensor


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)


class Trainer:
    def __init__(self):
        pass


def train(model, input_tensor, target_tensor, criterion, optimizer):
    hidden = model.init_hidden(input_tensor.size(0))
    optimizer.zero_grad()
    loss = 0
    for i in range(input_tensor.size(1)):
        output, hidden = model(input_tensor[:, i], hidden)
        loss += criterion(output.view(input_tensor.size(0), -1),
                          target_tensor[:, i])
    loss.backward()
    optimizer.step()

    return loss.item() / input_tensor.size(1)


def predict(model, input_tensor):
    hidden = model.init_hidden(input_tensor.size(0))
    for i in range(input_tensor.size(1)):
        output, hidden = model(input_tensor[:, i], hidden)
    return output


def predict_topk(model, input_tensor, k=5):
    output = predict(model, input_tensor)
    probabilities = nn.functional.softmax(output[0], dim=0)
    topk_probabilities, topk_indices = probabilities.topk(k)
    return topk_indices.tolist(), topk_probabilities.tolist()


data = ['select id, name from customer\0',
        'select id from customer\0',
        'select name from customer\0']

# 定義超參數
hidden_size = 128
n_epochs = 1000

# 創建模型實例
char_dict = CharDict()
model = CharRNN(len(char_dict.char_to_index), hidden_size, len(char_dict.char_to_index))

# 定義標準和優化器
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 執行多次訓練迭代
input_tensor, target_tensor = char_dict.create_train_data(data)
for epoch in range(n_epochs):
    loss = train(model, input_tensor, target_tensor, criterion, optimizer)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss}')

# 推斷句子 "select id" 的下一個單詞
test_sentence = "select id from customer"
test_input = [char_dict.char_to_index[char] for char in test_sentence]
test_input_tensor = torch.tensor([test_input])
topk_indices, topk_probabilities = predict_topk(model, test_input_tensor)

# 輸出結果
for i in range(len(topk_indices)):
    char = char_dict.index_to_char[topk_indices[i]]
    probability = topk_probabilities[i]
    print(f'{i + 1}. {char} ({probability:.2f})')
