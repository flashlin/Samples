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


class LstmModel(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_size, output_vocab_size):
        super().__init__()
        encoder_dim = hidden_size * 2
        decoder_dim = hidden_size
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=encoder_dim, num_heads=4)
        # self.softmax = nn.Softmax(-1)
        # self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.encoder = nn.LSTM(input_size=embedding_dim,
                               hidden_size=hidden_size,
                               num_layers=3,
                               batch_first=True,
                               bidirectional=True)
        self.decoder = nn.LSTM(input_size=encoder_dim + 1,
                               hidden_size=decoder_dim,
                               num_layers=3 * 2,
                               batch_first=True)
        # self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.output_linear = nn.Linear(decoder_dim, output_vocab_size)

    def forward(self, x, hidden=None):
        """
        :param x: [batch, n_sequence, input_vocab_size]
        :param hidden:
        :return:
        """
        embedded = self.embedding(x)
        hidden = Variable(torch.zeros(1, 1, self.hidden_size)) if None else hidden
        lstm_output, hidden = self.encoder(embedded, hidden)
        attention_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        decoder_input = torch.cat((x.unsqueeze(-1), attention_output), dim=-1)
        decoder_output, _ = self.decoder(decoder_input, hidden)
        output_x = self.output_linear(decoder_output)
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
        num_epochs = 600
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
        hat_y, _ = self.model(input_sequence)
        hat_y = hat_y.squeeze(0).argmax(1)
        return hat_y

        # output_sequence = []
        # with torch.no_grad():
        #     hidden = None
        #     for i in range(10):  # 預測輸出序列長度為 10
        #         output, hidden = self.model(input_sequence, hidden)
        #         _, predicted_indices = torch.max(output, dim=-1)
        #         last_output = predicted_indices[-1].item()
        #         output_sequence.append(last_output)
        #         # 將預測的輸出添加到輸入序列中，用於下一個時間步的預測
        #         new_input = output_sequence[-3:]
        #         input_sequence = torch.tensor(new_input, dtype=torch.long)
        # return output_sequence

m = Seq2SeqModel()
m.train()
output = m.infer([1, 2, 3])
print("Predicted Output Sequence:", output)
