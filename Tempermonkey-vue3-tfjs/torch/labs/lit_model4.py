import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from common.io import info
from ml.lit import start_train, CosineWarmupScheduler, BaseLightning
from labs.prepare import create_data_loader
from utils.linq_tokenizr import LINQ_VOCAB_SIZE
from utils.tsql_tokenizr import TSQL_VOCAB_SIZE
import random

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, inp, hidden):
        embedded = self.embedding(inp).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=500):
        super(AttnDecoderRNN, self).__init__()
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

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class LitMachineTranslation(BaseLightning):
    def __init__(self, src_vocab_size=LINQ_VOCAB_SIZE, trg_vocab_size=TSQL_VOCAB_SIZE,
                 hidden_size=256, device=None):
        super().__init__(device=device)
        self.max_length = 500
        self.lr_scheduler = None
        self.pad_token = 0
        self.sos_token = 1
        self.eos_token = 2
        self.encoder = EncoderRNN(src_vocab_size, hidden_size, self._device).to(self._device)
        self.encoder_hidden = self.encoder.initHidden()
        self.decoder = AttnDecoderRNN(hidden_size, trg_vocab_size, device=self._device, dropout_p=0.1).to(self._device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        train_loader, val_loader = create_data_loader()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.teacher_forcing_ratio = 0.5
        self.loss = 0

    def forward(self, x):
        src, trg = x

        input_length = src.size(0)
        target_length = trg.size(0)
        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self._device)
        encoder_hidden = self.encoder_hidden

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(src[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]
        decoder_input = torch.tensor([[self.sos_token]], device=self._device)
        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                info(f" d1 {decoder_output=} {trg[di]=} {di=}")
                loss += self.criterion(decoder_output, trg[di])
                decoder_input = trg[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                info(f" d2 {decoder_output=} {trg=}")
                loss += self.criterion(decoder_output, trg[di])
                if decoder_input.item() == self.eos_token:
                    break
        self.loss = loss
        return decoder_output

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        # optimizer = optim.Adam(self.parameters(), lr=0.0005)
        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=50, max_iters=80
        )
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        return self.loss


if __name__ == '__main__':
    start_train(LitMachineTranslation, device='cpu')
