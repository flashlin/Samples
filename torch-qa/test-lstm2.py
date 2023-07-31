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
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, x.size()[0], -1)
        output = embedded  # batch*seq*feature
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        # to(device)
        return result


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, hidden):
        output = self.embedding(x).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, use_gpu):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result


class Encoder2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Encoder2, self).__init__()
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


class Decoder2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder2, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x, hidden):
        print(f"de {x.shape=}")
        output = self.embedding(x)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, use_gpu):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        return result


class LstmModel(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, output_vocab_size,
                 sos_index=0, eos_index=1):
        super().__init__()
        self.sos_index = sos_index
        self.eos_index = eos_index
        decoder_dim = hidden_size
        self.hidden_size = hidden_size
        self.encoder = Encoder2(input_size=input_vocab_size,
                                hidden_size=hidden_size,
                                num_layers=3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)
        self.decoder_num_layers = 3 * 2
        self.decoder = Decoder2(input_size=hidden_size,
                                hidden_size=decoder_dim,
                                output_size=output_vocab_size,
                                num_layers=self.decoder_num_layers
                                )
        self.output_linear = nn.Linear(decoder_dim, output_vocab_size)
        self.fn_loss = nn.NLLLoss()
        self.decoder_dim = decoder_dim

    def forward(self, x, target=None):
        """
        :param x: [batch, n_sequence, input_vocab_size]
        :param target:
        :return:
        """
        encoder_output, encoder_hidden = self.encoder(x, None)

        if target is not None:
            decoder_hidden = encoder_hidden
            target_length = target.size(1)
            output_sequence = []
            for di in range(target_length):
                decoder_input = target[:, di].unsqueeze(1)
                h_o, c_o = decoder_hidden
                print(f"{c_o.shape=}")
                print(f"{encoder_output.shape=}")
                attention_output, _ = self.attention(c_o, encoder_output, encoder_output)
                decoder_input = torch.cat((decoder_input, attention_output), dim=-1)

                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                output_sequence.append(decoder_output)
                # # topv, topi = decoder_output.data.topk(1)
                # # ni = topi[0][0]
                # predict_index = output_x
                # output_sequence.append(predict_index)
                #
                # if target is None and predict_index == self.sos_index:
                #     break
                #
                # # 抓取最後輸出
                # if target is None:
                #     targ = torch.as_tensor([predict_index], dtype=torch.long)
                # else:
                #     targ = target[di]
                # decoder_input = targ.unsqueeze(0)
                # loss += self.fn_loss(predict_index, targ)
            output_sequence = torch.cat(output_sequence, dim=1)
            return output_sequence
        return self.infer(x, encoder_output, encoder_hidden)

    def infer(self, x, encoder_output, encoder_hidden):
        max_target_length = 100
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[self.sos_index]])  # Start of sequence token
        decoder_input = decoder_input.to(x.device)

        generated_sequence = []
        for _ in range(max_target_length):
            context, _ = self.attention(encoder_output, decoder_hidden)
            decoder_input = torch.cat((decoder_input, context), dim=-1)

            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            predicted_token = decoder_output.argmax(dim=-1)
            generated_sequence.append(predicted_token.item())

            if predicted_token == self.eos_index:
                break

        generated_sequence = torch.tensor(generated_sequence).unsqueeze(0)
        return generated_sequence

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

max_seq_len = 2


def prepare_train_data(data):
    for input, label in data:
        # input = pad_list(input, max_len=max_seq_len)
        # label = pad_list(label, max_len=max_seq_len)
        inputs = overlap_split_list(input, split_length=max_seq_len, overlap=1)
        labels = overlap_split_list(label, split_length=max_seq_len, overlap=1)
        for new_input, new_label in zip(inputs, labels):
            yield new_input, new_label


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
        num_epochs = 600
        optimizer = self.optimizer
        model = self.model
        model.train()
        best_loss = 1000
        for epoch in range(num_epochs):
            total_loss = 0
            for inputs, labels in loader:
                optimizer.zero_grad()
                padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
                outputs, hidden = model(padded_inputs, labels)
                outputs = outputs.view(-1, self.output_vocab_size)
                labels = labels.view(-1)
                # 交叉熵損失函數 nn.CrossEntropyLoss() 需要 labels 的形狀是 [batch_size], 而不是 [batch_size, sequence_length]
                loss = self.criterion(outputs, labels)
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
        input_sequence = torch.tensor(input_seq, dtype=torch.long)
        hat_y, _ = self.model(input_sequence)
        hat_y = hat_y.squeeze(0).argmax(1)
        return hat_y
        # output_sequence = []
        # count = 0
        # with torch.no_grad():
        #     hidden = None
        #     for i in range(10):  # 預測輸出序列長度為 10
        #         output, hidden = self.model(input_sequence, hidden)
        #         _, predicted_indices = torch.max(output, dim=-1)
        #         if count == 0:
        #             output_sequence.extend(predicted_indices.tolist())
        #         else:
        #             last_output = predicted_indices[-1].item()
        #             output_sequence.append(last_output)
        #         count += 1
        #         # 將預測的輸出添加到輸入序列中，用於下一個時間步的預測
        #         new_input = output_sequence[-3:]
        #         input_sequence = torch.tensor(new_input, dtype=torch.long)
        # return output_sequence


m = Seq2SeqModel()
m.load_model()
m.train()
output = m.infer([0, 2, 3, 4, 1])
print("Predicted Output Sequence:", output)
