import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl

from common.io import info


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LitEncoder(pl.LightningModule):
    def __init__(self, vocab_len, embedding_dim, encoder_hidden_dim, n_layers=1, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_len, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, n_layers, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_batch):
        embedded = self.dropout(self.embedding(input_batch))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class LitAttention(pl.LightningModule):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()

        # The input dimension will the the concatenation of
        # encoder_hidden_dim (hidden) and  decoder_hidden_dim(encoder_outputs)
        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)

        # We need source len number of values for n batch as the dimension
        # of the attention weights. The attn_hidden_vector will have the
        # dimension of [source len, batch size, decoder hidden dim]
        # If we set the output dim of this Linear layer to 1 then the
        # effective output dimension will be [source len, batch size]
        self.attn_scoring_fn = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, decoder hidden dim]
        src_len = encoder_outputs.shape[0]

        # We need to calculate the attn_hidden for each source words.
        # Instead of repeating this using a loop, we can duplicate
        # hidden src_len number of times and perform the operations.
        hidden = hidden.repeat(src_len, 1, 1)

        # Calculate Attention Hidden values
        attn_hidden = torch.tanh(self.attn_hidden_vector(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate the Scoring function. Remove 3rd dimension.
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(2)

        # The attn_scoring_vector has dimension of [source len, batch size]
        # Since we need to calculate the softmax per record in the batch
        # we will switch the dimension to [batch size,source len]
        attn_scoring_vector = attn_scoring_vector.permute(1, 0)

        # Softmax function for normalizing the weights to
        # probability distribution
        return F.softmax(attn_scoring_vector, dim=1)


class LitOneStepDecoder(pl.LightningModule):
    def __init__(self, input_output_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, attention, dropout_prob=0.5):
        super().__init__()

        self.output_dim = input_output_dim
        self.attention = attention

        self.embedding = nn.Embedding(input_output_dim, embedding_dim)

        # Add the encoder_hidden_dim and embedding_dim
        self.rnn = nn.GRU(encoder_hidden_dim + embedding_dim, decoder_hidden_dim)
        # Combine all the features for better prediction
        self.fc = nn.Linear(encoder_hidden_dim + decoder_hidden_dim + embedding_dim, input_output_dim)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, hidden, encoder_outputs):
        # Add the source len dimension
        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        # Calculate the attention weights
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        # We need to perform the batch wise dot product.
        # Hence need to shift the batch dimension to the front.
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Use PyTorch's bmm function to calculate the
        # weight W.
        W = torch.bmm(a, encoder_outputs)

        # Revert the batch dimension.
        W = W.permute(1, 0, 2)

        # concatenate the previous output with W
        rnn_input = torch.cat((embedded, W), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        # Remove the sentence length dimension and pass them to the Linear layer
        predicted_token = self.fc(torch.cat((output.squeeze(0), W.squeeze(0), embedded.squeeze(0)), dim=1))

        return predicted_token, hidden, a.squeeze(1)


class LitDecoder(pl.LightningModule):
    def __init__(self, one_step_decoder, device):
        super().__init__()
        self.one_step_decoder = one_step_decoder
        self._device = device

    def forward(self, target, encoder_outputs, hidden, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        trg_len = target.shape[0]
        trg_vocab_size = self.one_step_decoder.output_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self._device)
        input = target[0, :]

        for t in range(1, trg_len):
            # Pass the encoder_outputs. For the first time step the
            # hidden state comes from the encoder model.
            output, hidden, a = self.one_step_decoder(input, hidden, encoder_outputs)
            outputs[t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = target[t] if teacher_force else top1

        return outputs


class LitEncodeDecoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        encoder_outputs, hidden = self.encoder(source)
        return self.decoder(target, encoder_outputs, hidden, teacher_forcing_ratio)


class LitTranslateModel(pl.LightningModule):
    def __init__(self, source_vocab_size, target_vocab_size, device, embedding_dim=256, hidden_dim = 1024):
        super().__init__()
        self.attention_model = LitAttention(embedding_dim, hidden_dim)
        self.encoder = LitEncoder(source_vocab_size, embedding_dim, hidden_dim)
        self.one_step_decoder = LitOneStepDecoder(target_vocab_size, embedding_dim, hidden_dim, hidden_dim, self.attention_model)
        self.decoder = LitDecoder(self.one_step_decoder, device)
        self.model = LitEncodeDecoder(self.encoder, self.decoder)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.log_softmax_fn = torch.nn.LogSoftmax(dim=1)

    def forward(self, source, target):
        return self.model(source, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self.model(x, y)
        # reshape the output
        output_dim = output.shape[-1]
        # Discard the first token as this will always be 0
        output = output[1:].view(-1, output_dim)
        # Discard the sos token from target
        trg = y[1:].view(-1)
        # Calculate the loss
        loss = self.criterion(output, trg)
        return loss

    def validation_step(self, batch, batch_idx):  # Optional for PyTorch Lightning
        self.eval()
        reactants, products = batch
        logits = self.forward(reactants, products)  # Sequence, batch, tokens
        logits = logits.permute(1, 2, 0)  # Now batch, tokens, sequence
        log_softmax = self.log_softmax_fn(logits)
        loss = self.criterion(log_softmax, products[1:].permute(1, 0))  # Skipping the start-char in the target
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


if __name__ == '__main__':
    # model = train(train_iterator, valid_iterator, source, target, epochs=25)
    #
    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'source': source.vocab,
    #     'target': target.vocab
    # }
    #
    # torch.save(checkpoint, 'nmt-model-gru-attention-25.pth')
    pass