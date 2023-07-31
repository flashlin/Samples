import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_utils import create_running_list, pad_list, overlap_split_list


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # in:(batch_size, seq_len) out:(batch_size, seq_len, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # in:(batch_size,seq_len,input_size)
        # out: (batch_size,seq_len, num_directions*hidden_size),
        #      (h_n:隱藏狀態(num_layers*num_directions, batch_size, hidden_size), c_n:最後一個時間步的細胞狀態)
        self.encoder = nn.LSTM(input_size=hidden_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

    def forward(self, x, encoder_hidden):
        embedded = self.embedding(x)
        encoder_output, encoder_hidden = self.encoder(embedded, encoder_hidden)
        return encoder_output, encoder_hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        # to(device)
        return result


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        output = self.embedding(x)
        # output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        return output, hidden

    def initHidden(self, use_gpu):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result


def r_trim(tensor):
    end_index = len(tensor) - 1
    last_value = tensor[end_index]
    while end_index >= 0 and tensor[end_index] == last_value:
        end_index -= 1
    result = tensor[:end_index + 2]
    return result


class LstmModel(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, output_vocab_size,
                 sos_index=0, eos_index=1):
        super().__init__()
        self.sos_index = sos_index
        self.eos_index = eos_index
        decoder_dim = hidden_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_size=input_vocab_size,
                               hidden_size=hidden_size * 2,
                               num_layers=3)
        # in:(batch_size, sequence_length, hidden_size)
        self.decoder_num_layers = 3 * 2
        self.decoder = Decoder(input_size=hidden_size,
                               hidden_size=decoder_dim * 2,
                               output_size=output_vocab_size,
                               num_layers=self.decoder_num_layers
                               )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2,
                                               num_heads=hidden_size)
        self.output_linear = nn.Linear(decoder_dim * 2, output_vocab_size)
        # self.fn_loss = nn.NLLLoss()
        self.fn_loss = nn.CrossEntropyLoss()
        self.decoder_dim = decoder_dim

    def forward(self, x, target=None):
        """
        :param x: [batch, n_sequence, input_vocab_size]
        :param target:
        :return:
        """
        # print(f"{x.shape=}")
        # print(f"{target.shape=}")
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

                # print(f"{decoder_input.shape=}")
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                decoder_output = self.exec_attention(batch_size, encoder_output, decoder_output)
                output_sequence.append(decoder_output)
                step_loss = self.fn_loss(decoder_output.squeeze(1), target[:, di])
                loss += step_loss
            output_sequence = torch.cat(output_sequence, dim=1)
            return output_sequence, loss
        return self.infer(x, encoder_output, encoder_hidden)

    def exec_attention(self, batch_size, encoder_output, decoder_output):
        decoder_output = decoder_output.transpose(0, 1)
        encoder_output = encoder_output.transpose(0, 1)
        encoder_output = encoder_output.view(-1, batch_size, self.decoder_dim * 2)
        # print(f"2 {decoder_output.shape=}")
        # print(f"2 {encoder_output.shape=}")
        attention_output, _ = self.attention(query=decoder_output,
                                             key=encoder_output,
                                             value=encoder_output)

        attention_output = attention_output.transpose(0, 1)
        attention_output = self.output_linear(attention_output[0])
        attention_output = F.log_softmax(attention_output, dim=-1)
        return attention_output

    def infer(self, x, encoder_output, encoder_hidden):
        max_target_length = 100
        decoder_hidden = encoder_hidden
        decoder_input = torch.as_tensor([self.sos_index]).unsqueeze(0)  # Start of sequence token
        decoder_input = decoder_input.to(x.device)

        generated_sequence = []
        for _ in range(max_target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            decoder_output = self.exec_attention(1, encoder_output, decoder_output)
            predicted_token = decoder_output.argmax(dim=-1)
            decoder_input = torch.cat((decoder_input, predicted_token.unsqueeze(0)), dim=-1)
            # print(f"{predicted_token=}")
            generated_sequence.append(predicted_token)
            if predicted_token[-1] == self.eos_index:
                break

        generated_sequence = torch.cat(generated_sequence)
        generated_sequence = r_trim(generated_sequence)
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


data = [
    ([0, 2, 3, 1], [0, 4, 5, 6, 1]),
    ([0, 2, 3, 4, 1], [0, 4, 7, 8, 9, 1]),
]


def prepare_train_data(data):
    for input, label in data:
        yield input, label


train_data = []
for input, label in prepare_train_data(data):
    train_data.append((input, label))

dataset = MyDataset(train_data)
loader = DataLoader(dataset, batch_size=1)


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

    def load_model(self):
        pth_file = './models/test3.pth'
        model = self.model
        if os.path.exists(pth_file):
            model.load_state_dict(torch.load(pth_file))

    def train(self):
        pth_file = './models/test3.pth'
        num_epochs = 200
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
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if total_loss < best_loss:
                best_loss = total_loss
            # 每 100 次迭代輸出一次訓練損失
            if epoch % 100 == 0:
                torch.save(model.state_dict(), pth_file)
                print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.5f}')

    def infer(self, input_seq):
        input_sequence = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
        hat_y, _ = self.model(input_sequence)
        return hat_y


m = Seq2SeqModel()
m.load_model()
# m.train()
output = m.infer([0, 2, 3, 4, 1])
print("Predicted Output Sequence:", output)
