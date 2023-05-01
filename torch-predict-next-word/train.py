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
        hidden_size = 128
        self.model_path = "./output/model.pth"
        self.n_epochs = 500
        self.char_dict = char_dict = CharDict()
        self.model = model = CharRNN(len(char_dict.char_to_index), hidden_size, len(char_dict.char_to_index))
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = SGD(model.parameters(), lr=0.01)

    def train(self, data):
        best_loss = float('inf')
        best_state_dict = None
        input_tensor, target_tensor = self.char_dict.create_train_data(data)
        for epoch in range(self.n_epochs):
            loss = self.train_loop(input_tensor, target_tensor)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}, Loss: {loss}')
            if loss < best_loss:
                best_loss = loss
                best_state_dict = self.model.state_dict()
        torch.save(best_state_dict, self.model_path)

    def infer(self, test_sentence):
        # 推斷句子 "select id" 的下一個單詞
        char_dict = self.char_dict
        test_input = [char_dict.char_to_index[char] for char in test_sentence]
        test_input_tensor = torch.tensor([test_input])
        topk_indices, topk_probabilities = self.predict_topk(test_input_tensor)
        # 輸出結果
        result = []
        for i in range(len(topk_indices)):
            char = char_dict.index_to_char[topk_indices[i]]
            probability = topk_probabilities[i]
            # print(f'{i + 1}. {char} ({probability:.2f})')
            result.append((char, probability))
        return result

    def train_loop(self, input_tensor, target_tensor):
        hidden = self.model.init_hidden(input_tensor.size(0))
        self.optimizer.zero_grad()
        loss = 0
        for i in range(input_tensor.size(1)):
            output, hidden = self.model(input_tensor[:, i], hidden)
            loss += self.criterion(output.view(input_tensor.size(0), -1), target_tensor[:, i])
        loss.backward()
        self.optimizer.step()
        return loss.item() / input_tensor.size(1)

    def predict_topk(self, input_tensor, k=5):
        output = self.predict(input_tensor)
        probabilities = nn.functional.softmax(output[0], dim=0)
        topk_probabilities, topk_indices = probabilities.topk(k)
        return topk_indices.tolist(), topk_probabilities.tolist()

    def predict(self, input_tensor):
        hidden = self.model.init_hidden(input_tensor.size(0))
        for i in range(input_tensor.size(1)):
            output, hidden = self.model(input_tensor[:, i], hidden)
        return output


trainer = Trainer()
data = ['select id, name from customer\0',
        'select id from customer\0',
        'select name from customer\0']
trainer.train(data)


results = trainer.infer("select birth")
for char, prob in results:
    print(f"'{char}' {prob=}")
