import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_running_list, pad_list, overlap_split_list
from tsql_tokenizr import tsql_tokenize


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim: int, vocab_size: int = 5000, dropout_p: float = 0.1):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.lut = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        :param x: (batch, L) L為batch中最長句子長度
        :return (batch, L, embed_dim)
        """
        x = self.lut(x) * math.sqrt(self.embed_dim)
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # in:(batch_size, seq_len) out:(batch_size, seq_len, hidden_size)
        # self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.embedding = PositionalEmbedding(hidden_size)
        # in:(batch_size,seq_len,input_size)
        # out: (batch_size,seq_len, num_directions*hidden_size),
        #      (h_n:隱藏狀態(num_layers*num_directions, batch_size, hidden_size), c_n:最後一個時間步的細胞狀態)
        self.encoder = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

    def forward(self, x, encoder_hidden):
        """
            :param x: (batch_size, seq_len)
            :param encoder_hidden:
        """
        embedded = self.embedding(x)
        encoder_output, encoder_hidden = self.encoder(embedded, encoder_hidden)
        return encoder_output, encoder_hidden


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        # output = F.relu(output)
        lstm_output, hidden = self.lstm(embedded, hidden)
        return lstm_output, hidden


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, output_vocab_size):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=num_heads)
        self.output_linear = nn.Linear(embed_dim, output_vocab_size)
        self.embed_dim = embed_dim

    def forward(self, query, key, value):
        """
            :param query:
            :param key:
            :param value:
            :return:
        """
        attention_output, hidden_state = self.attention(query=query,
                                                        key=key,
                                                        value=value)
        return attention_output, hidden_state

    def exec_forward(self, batch_size, encoder_output, decoder_output):
        decoder_output = decoder_output.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)
        encoder_output = encoder_output.view(-1, batch_size, self.embed_dim)
        attention_output, hidden_state = self.forward(query=decoder_output,
                                                      key=encoder_output,
                                                      value=encoder_output)
        attention_output = attention_output.transpose(0, 1)
        attention_output = self.output_linear(attention_output[0])
        attention_output = F.log_softmax(attention_output, dim=-1)
        return attention_output, hidden_state


def trim_right(tensor):
    end_index = len(tensor) - 1
    last_value = tensor[end_index]
    while end_index >= 0 and tensor[end_index] == last_value:
        end_index -= 1
    result = tensor[:end_index + 2]
    return result


class LstmModel(nn.Module):
    def __init__(self, input_vocab_size, hidden_size, output_vocab_size, sos_index=1, eos_index=2):
        super().__init__()
        self.sos_index = sos_index
        self.eos_index = eos_index
        self.encoder_dim = encoder_dim = hidden_size * 2
        self.encoder_layers = encoder_num_layers = 3
        self.encoder = Encoder(input_size=input_vocab_size,
                               hidden_size=encoder_dim,
                               num_layers=encoder_num_layers)
        # in:(batch_size, sequence_length, hidden_size)
        self.decoder_dim = decoder_dim = encoder_dim
        self.decoder_num_layers = encoder_num_layers * 2
        self.decoder = Decoder(input_size=input_vocab_size,
                               hidden_size=decoder_dim,
                               output_size=output_vocab_size,
                               num_layers=self.decoder_num_layers
                               )
        self.attention = MultiHeadAttention(embed_dim=hidden_size * 2,
                                            num_heads=hidden_size,
                                            output_vocab_size=output_vocab_size)
        # self.fn_loss = nn.NLLLoss()
        self.fn_loss = nn.CrossEntropyLoss()

    def forward(self, x, target=None):
        """
        :param x: [batch, n_sequence, input_vocab_size]
        :param target:
        :return:
        """
        # print(f"{x.shape=}")
        batch_size = x.size(0)
        encoder_output, encoder_hidden = self.encoder(x, None)

        if target is not None:
            decoder_hidden = encoder_hidden
            target_length = target.size(1)
            # print(f"{target_length=}")
            output_sequence = []
            loss = 0
            for di in range(target_length):
                decoder_input = target[:, di].unsqueeze(0)
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_output = self.exec_attention(batch_size, encoder_output, decoder_output)
                predicted_token = decoder_output.argmax(dim=-1)
                output_sequence.append(predicted_token)
                step_loss = self.fn_loss(decoder_output.squeeze(1), target[:, di])
                loss += step_loss
            output_sequence = torch.cat(output_sequence)
            return output_sequence, loss
        return self.infer(x, encoder_output, encoder_hidden)

    def exec_attention(self, batch_size, encoder_output, decoder_output):
        attention_output, _ = self.attention.exec_forward(
            batch_size=batch_size,
            encoder_output=encoder_output,
            decoder_output=decoder_output)
        return attention_output

    def infer(self, x, encoder_output, encoder_hidden):
        max_target_length = 100
        decoder_hidden = encoder_hidden
        decoder_input = torch.as_tensor([self.sos_index]).unsqueeze(0)  # Start of sequence token
        decoder_input = decoder_input.to(x.device)

        generated_sequence = []
        generated_length = 0
        while generated_length < max_target_length:
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.exec_attention(1, encoder_output, decoder_output)
            predicted_token = decoder_output.argmax(dim=-1)
            predicted_len = len(predicted_token)
            decoder_input = torch.cat((decoder_input, predicted_token.unsqueeze(0)), dim=-1)
            generated_sequence.append(predicted_token)
            if predicted_token[-1] == self.eos_index:
                break
            generated_length += predicted_len

        generated_sequence = torch.cat(generated_sequence)
        generated_sequence = trim_right(generated_sequence)
        return generated_sequence, None


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


raw_data = [
    ("select id from customer",
     """{"type":"select","cols":["id as id"],"fromCause":"customer as customer"}"""),
    ("select id, name from customer",
     """{"type":"select","cols":["id as id","name as name"],"fromCause":"customer as customer"}"""),
    ("select id1 as id, name from customer",
     """{"type":"select","cols":["id1 as id","name as name"],"fromCause":"customer as customer"}"""),
    ("select id as id, name1 as name from customer",
     """{"type":"select","cols":["id as id","name1 as name"],"fromCause":"customer as customer"}"""),
]


def str_to_id(text: str) -> list[int]:
    ascii_codes = [ord(ch) for ch in text]
    return ascii_codes


def id_to_str(values: list[int]) -> str:
    ascii_codes = [chr(code) for code in values]
    return ''.join(ascii_codes)


def sql_to_id(sql: str) -> list[int]:
    tokens = tsql_tokenize(sql)
    sql_tokens = [token.text for token in tokens]
    sql_tokens2 = []
    for n in range(len(sql_tokens)):
        sql_tokens2.append(sql_tokens[n])
        if n < len(sql_tokens) - 1:
            sql_tokens2.append(' ')
    sql2 = ''.join(sql_tokens2)
    return str_to_id(sql2)


train_data = []

start_index = 1
end_index = 2


def prepare_train_data(data):
    for sql, label in data:
        input_values = [start_index] + sql_to_id(sql) + [end_index]
        label_values = [start_index] + str_to_id(label) + [end_index]
        train_data.append((input_values, label_values))


def my_collate(batch):
    inputs = []
    labels = []
    for sentence, label in batch:
        input_value = sentence
        label_value = label
        inputs.append(input_value)
        labels.append(label_value)
    return inputs, labels


prepare_train_data(raw_data)

dataset = MyDataset(train_data)
loader = DataLoader(dataset, batch_size=1)


class Seq2SeqModel:
    def __init__(self):
        vocab_size = 128 + 3
        self.model = model = LstmModel(input_vocab_size=vocab_size,
                                       hidden_size=64,
                                       output_vocab_size=vocab_size)
        # 定義損失函數和優化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)

    def load_model(self):
        pth_file = './models/test3.pth'
        model = self.model
        if os.path.exists(pth_file):
            model.load_state_dict(torch.load(pth_file))

    def train(self):
        pth_file = './models/test3.pth'
        num_epochs = 100
        optimizer = self.optimizer
        model = self.model
        model.train()
        best_loss = 100
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in loader:
                optimizer.zero_grad()
                padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
                outputs, loss = model(padded_inputs, labels)
                # print(f"{outputs=}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if total_loss < best_loss:
                best_loss = total_loss
            # 每 100 次迭代輸出一次訓練損失
            if epoch % 10 == 0:
                torch.save(model.state_dict(), pth_file)
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}')

    def infer(self, input_seq):
        input_sequence = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
        hat_y, _ = self.model(input_sequence)
        return hat_y


m = Seq2SeqModel()
m.load_model()
m.train()


print(f"----------------------------------")

sql = "select name from c"
sql_value = sql_to_id(sql)
output = m.infer(sql_value)
rc = id_to_str(output)
print("Predicted Output:", output)
print("Predicted Output Sequence:", rc)