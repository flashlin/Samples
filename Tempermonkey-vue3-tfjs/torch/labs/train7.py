import random

import torch
from torch import nn
import torch.nn.functional as F
from common.io import info
from ml.lit import BaseLightning, start_train
from prepare7 import Linq2TSqlDataset
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tokenizr import VOCAB_SIZE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE

MAX_LENGTH = 500

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1) #unsqueeze(0)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, input_length):
        return torch.zeros(1, input_length, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    # return mask

class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size=LINQ_VOCAB_SIZE + 1, tgt_vocab_size=TSQL_VOCAB_SIZE, device='cpu'):
        super().__init__()
        hidden_size = 3
        self.device = device
        self.encoder = EncoderRNN(src_vocab_size, hidden_size, device)
        self.decoder = AttnDecoderRNN(hidden_size, tgt_vocab_size, dropout_p=0.1, device=device)
        self.criterion = nn.NLLLoss()
        self.max_length = 500
        self.sos_token = 1
        self.eos_token = 2
        self.teacher_forcing_ratio = 0.4

    def forward(self, x, y):
        input_length = x.size(0)
        target_length = y.size(0)
        encoder_hidden = self.encoder.initHidden(input_length)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(x[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_input = torch.tensor([[self.sos_token]], device=self.device)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        decoder_hidden = encoder_hidden
        loss = 0
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, y[di])
                decoder_input = y[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                loss += self.criterion(decoder_output, y[di])
                if decoder_input.item() == self.eos_token:
                    break
        loss1 = loss / target_length
        return loss1

class LitSeq2Seq(BaseLightning):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=512, hidden_dim=3, dropout=0.2):
        super().__init__()
        self.n_tokens = vocab_size
        self.model = TranslationModel()
        ds = Linq2TSqlDataset('../output/linq-sample.csv')
        train_loader, val_loader = ds.create_dataloader()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = val_loader
        self.criterion = nn.NLLLoss()
        self.src_mask = generate_square_subsequent_mask(35)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self.model(x, y)

    def validation_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.training_step(batch, batch_idx)

    def _calculate_loss(self, batch, mode="train"):
        y_hat, y = batch
        info(f" {self.n_tokens=} {y_hat.shape=} {y.shape=}")
        info(f" {y_hat.view(-1, self.n_tokens).shape=} {y.shape=}")
        self.criterion(y_hat.view(-1, self.n_tokens), y)
        loss = self.loss_compute(y_hat, y, self.tgt_vocab_size)
        self.log("%s_loss" % mode, loss)
        # y_hat, y = batch
        # acc = self._compute_accuracy(y_hat, y)
        # self.log("%s_acc" % mode, acc)
        return loss

    # def _compute_accuracy(self, y_hat, y):
    #     return accuracy(y_hat, y)

    def infer(self):
        # pred = model(seq_in)
        pred = 1.0
        # pred = to_prob(F.softmax(pred).data[0].numpy())  # softmax 後轉成機率分佈
        # char = np.random.choice(chars, p=pred)  # 依機率分佈選字
        pass


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # start_train(LitSeq2Seq, device='cpu', src_vocab_size=LINQ_VOCAB_SIZE, tgt_vocab_size=TSQL_VOCAB_SIZE)
    info(f" {LINQ_VOCAB_SIZE=} {TSQL_VOCAB_SIZE=}")
    start_train(LitSeq2Seq, device='cpu')

if __name__ == "__main__":
    main()