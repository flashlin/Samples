from datetime import datetime
from sql_repo import SqlRepo
from torch.optim import SGD
import torch
import torch.nn as nn
import os
import string
from flask import Flask, request


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

    def create_infer_data(self, sentence):
        test_input = [self.char_to_index[char] for char in sentence]
        test_input_tensor = torch.tensor([test_input])
        return test_input_tensor

    def decode_index(self, value):
        return self.index_to_char[value]


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
        self.n_epochs = 200
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

    def infer(self, test_sentence, k=5):
        # 推斷句子 "select id" 的下一個單詞
        char_dict = self.char_dict
        test_input_tensor = char_dict.create_infer_data(test_sentence)
        top_k_indices, top_k_probabilities = self.predict_top_k(test_input_tensor, k)
        # 輸出結果
        result = []
        for i in range(len(top_k_indices)):
            c_char = char_dict.index_to_char[top_k_indices[i]]
            probability = top_k_probabilities[i]
            # print(f'{i + 1}. {char} ({probability:.2f})')
            result.append((c_char, round(probability, 2)))
        return result

    def infer_sentence(self, sentence, k=5):
        top_k_list = self.infer(sentence, k)
        result = []
        for c_char, probability in top_k_list:
            if probability < 0.5:
                break
            next_chars = self.internal_infer_sentence(sentence, [c_char])
            next_chars = ''.join(next_chars)
            result.append({
                "next_words": next_chars,
                "probability": probability
            })
        return result

    def internal_infer_sentence(self, sentence, next_chars):
        if next_chars[-1] == '\0':
            return next_chars[:-1]
        next_char, _ = self.infer(sentence + ''.join(next_chars), k=1)[0]
        next_chars.append(next_char)
        return self.internal_infer_sentence(sentence, next_chars)

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

    def predict_top_k(self, input_tensor, k=5):
        output = self.predict(input_tensor)
        probabilities = nn.functional.softmax(output[0], dim=0)
        topk_probabilities, topk_indices = probabilities.topk(k)
        return topk_indices.tolist(), topk_probabilities.tolist()

    def predict(self, input_tensor):
        hidden = self.model.init_hidden(input_tensor.size(0))
        for i in range(input_tensor.size(1)):
            output, hidden = self.model(input_tensor[:, i], hidden)
        return output

sql_repo = SqlRepo()
trainer = Trainer()
app = Flask(__name__)


@app.route('/addsql', methods=['POST'])
def add_sql():
    data = request.get_json()
    input_sql = data['sql']
    _add_sql(input_sql)

@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    input_sentence = data['input']
    return _infer(input_sentence)

def _add_sql(input_sql):
    try:
        sql_repo.execute('insert into _sqlHistory(sql) values(?)', (input_sql,))
    except Exception as e:
        id = sql_repo.query_first('select id from _sqlHistory where sql=?', (input_sql,))[0]
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql_repo.execute('update _sqlHistory set createOn=? where id=?', (current_time,id,))
        sql_repo.commit()
    rc = sql_repo.query_first('select id from _sqlHistory order by createOn desc LIMIT 1 OFFSET 999')
    if rc is not None:
        sql_repo.execute('delete _sqlHistory where id < ?', (rc[0],))
    sql_repo.commit()
    data = []
    for row in sql_repo.query('select sql from _sqlHistory'):
        data.append(row[0] + '\0')
    trainer.train(data)

def _infer(input_sentence):
    top_k = trainer.infer_sentence(input_sentence)
    return {'top_k': top_k}


def test1():
    trainer = Trainer()
    data = ['select id, name from customer\0',
            'select id from customer\0',
            'select name from customer\0']
    data = ['select id from customer\0']
    trainer.train(data)

    results = trainer.infer("select birth")
    for ch, prob in results:
        print(f"'{ch}' {prob=}")

    print("-----------------")
    print("select id")
    results = trainer.infer_sentence("select id")
    for item in results:
        print(f"'{item['next_words']}' {item['probability']=}")

def test2():
    _add_sql('select id,name from customer\0')
    top_k = _infer('select name ')['top_k']
    for item in top_k:
        print(f"'{item['next_words']}' {item['probability']=}")

if __name__ == '__main__':
    sql_repo.execute('''create table IF NOT EXISTS _sqlHistory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sql TEXT NOT NULL UNIQUE,
        createOn DATETIME DEFAULT CURRENT_TIMESTAMP
    )''',)
    app.run(host="0.0.0.0",port=8000)
    #_add_sql('select id from customer\0')
    